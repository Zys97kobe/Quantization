import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.dirname(scriptDir);

function usage() {
  return `用法：
  ./scripts/generate_monthly_report --month YYYY-MM
  ./scripts/generate_monthly_report

参数：
  --month YYYY-MM       指定统计月份；不传时默认生成上个月
  --source PATH         收益 CSV，默认 data/paper/daily_returns.csv
  --output-dir PATH     输出目录，默认 reports/monthly
  --preview             同时保存两张工作表的 PNG 预览
  -h, --help            显示帮助`;
}

function previousMonthInShanghai() {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai",
    year: "numeric",
    month: "2-digit",
  }).formatToParts(new Date());
  let year = Number(parts.find((part) => part.type === "year").value);
  let month = Number(parts.find((part) => part.type === "month").value) - 1;
  if (month === 0) {
    year -= 1;
    month = 12;
  }
  return `${year}-${String(month).padStart(2, "0")}`;
}

function parseArgs(argv) {
  const options = {
    month: previousMonthInShanghai(),
    source: path.join(projectRoot, "data/paper/daily_returns.csv"),
    outputDir: path.join(projectRoot, "reports/monthly"),
    preview: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "-h" || arg === "--help") {
      console.log(usage());
      process.exit(0);
    } else if (arg === "--preview") {
      options.preview = true;
    } else if (["--month", "--source", "--output-dir"].includes(arg)) {
      if (!argv[i + 1]) throw new Error(`${arg} 缺少参数值`);
      const value = argv[i + 1];
      i += 1;
      if (arg === "--month") options.month = value;
      if (arg === "--source") options.source = path.resolve(value);
      if (arg === "--output-dir") options.outputDir = path.resolve(value);
    } else {
      throw new Error(`未知参数：${arg}\n\n${usage()}`);
    }
  }
  if (!/^\d{4}-(0[1-9]|1[0-2])$/.test(options.month)) {
    throw new Error(`月份格式错误：${options.month}，应为 YYYY-MM`);
  }
  return options;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  if (rows.length < 2) throw new Error("收益 CSV 没有数据");
  const headers = rows[0];
  return rows.slice(1).filter((r) => r.some((value) => value !== "")).map((r) =>
    Object.fromEntries(headers.map((header, index) => [header, r[index] ?? ""]))
  );
}

function excelColumn(index) {
  let n = index + 1;
  let result = "";
  while (n > 0) {
    const remainder = (n - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    n = Math.floor((n - 1) / 26);
  }
  return result;
}

const options = parseArgs(process.argv.slice(2));
const [yearText, monthText] = options.month.split("-");
const year = Number(yearText);
const monthNumber = Number(monthText);
const monthStart = `${options.month}-01`;
const nextMonth = monthNumber === 12
  ? `${year + 1}-01-01`
  : `${year}-${String(monthNumber + 1).padStart(2, "0")}-01`;
const daysInMonth = new Date(Date.UTC(year, monthNumber, 0)).getUTCDate();

const rawRows = parseCsv(await fs.readFile(options.source, "utf8"));
const lastByDate = new Map();
for (const row of rawRows) {
  if (row.date >= monthStart && row.date < nextMonth) lastByDate.set(row.date, row);
}
const monthRows = [...lastByDate.values()];
const priorRow = rawRows.filter((row) => row.date < monthStart).at(-1);
if (!priorRow) throw new Error(`${options.month} 之前没有月初权益基准记录`);
if (monthRows.length === 0) throw new Error(`${options.month} 没有可生成的收益记录`);

const requiredFields = ["date", "cash", "equity", "daily_pnl", "daily_return", "total_return", "phase"];
for (const field of requiredFields) {
  if (!(field in monthRows[0])) throw new Error(`收益 CSV 缺少字段：${field}`);
}

const startEquity = Number(priorRow.equity);
const endEquity = Number(monthRows.at(-1).equity);
const pnlSum = monthRows.reduce((sum, row) => sum + Number(row.daily_pnl), 0);
if (Math.abs((endEquity - startEquity) - pnlSum) > 0.01) {
  throw new Error("月度盈亏与日终权益变化不一致，请先检查 daily_returns.csv");
}

const defaultArtifactPath = path.join(
  os.homedir(),
  ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs",
);
const artifactPath = process.env.ARTIFACT_TOOL_MODULE || defaultArtifactPath;
let SpreadsheetFile;
let Workbook;
try {
  ({ SpreadsheetFile, Workbook } = await import(pathToFileURL(artifactPath).href));
} catch (error) {
  throw new Error(`无法加载表格组件 ${artifactPath}：${error.message}`);
}

const workbook = Workbook.create();
const calendar = workbook.worksheets.add(`${monthNumber}月日历`);
const detail = workbook.worksheets.add("每日明细");
const colors = {
  navy: "#17365D", blue: "#2F75B5", lightBlue: "#D9EAF7", paleBlue: "#EEF5FB",
  white: "#FFFFFF", text: "#243447", muted: "#687386", border: "#C9D4E2",
  weekend: "#F1F3F5", profit: "#C00000", profitFill: "#FCE8E6",
  loss: "#008000", lossFill: "#E8F5E9", neutral: "#F8FAFC",
};
const border = {
  top: { style: "thin", color: colors.border }, bottom: { style: "thin", color: colors.border },
  left: { style: "thin", color: colors.border }, right: { style: "thin", color: colors.border },
};

detail.mergeCells("A1:L1");
detail.getRange("A1").values = [[`${year}年${monthNumber}月模拟盘每日收益明细`]];
detail.getRange("A1:L1").format = {
  fill: colors.navy, font: { bold: true, color: colors.white, size: 18 },
  horizontalAlignment: "center", verticalAlignment: "center", rowHeight: 30,
};
const firstDataRow = 7;
const lastDataRow = firstDataRow + monthRows.length - 1;
detail.getRange("A3:L3").values = [[
  "月初权益（元）", startEquity, "月末权益（元）", null, "月度净收益（元）", null,
  "月度收益率", null, "盈利日", null, "亏损日", null,
]];
detail.getRange("A3:L3").format = {
  fill: colors.paleBlue, font: { bold: true, color: colors.text }, borders: border,
  verticalAlignment: "center", rowHeight: 23,
};
detail.getRange("D3").formulas = [[`=E${lastDataRow}`]];
detail.getRange("F3").formulas = [["=D3-B3"]];
detail.getRange("H3").formulas = [["=D3/B3-1"]];
detail.getRange("J3").formulas = [[`=COUNTIF(F${firstDataRow}:F${lastDataRow},\">0\")`]];
detail.getRange("L3").formulas = [[`=COUNTIF(F${firstDataRow}:F${lastDataRow},\"<0\")`]];
for (const cell of ["B3", "D3", "F3"]) detail.getRange(cell).format.numberFormat = "#,##0.00;[Green]-#,##0.00";
detail.getRange("H3").format.numberFormat = "0.00%;[Green]-0.00%";
detail.mergeCells("A5:L5");
detail.getRange("A5").values = [["口径：每个自然日取 daily_returns.csv 中最后一条账户快照；当日盈亏相对上一交易日最终权益。"]];
detail.getRange("A5:L5").format = {
  fill: colors.neutral, font: { italic: true, color: colors.muted, size: 10 }, rowHeight: 20,
};
detail.getRange("A6:L6").values = [[
  "日期", "星期", "日终阶段", "现金（元）", "日终权益（元）", "当日盈亏（元）",
  "当日收益率", "累计收益率", "当前持仓", "新开仓", "已结算交易", "备注",
]];
detail.getRange("A6:L6").format = {
  fill: colors.blue, font: { bold: true, color: colors.white }, horizontalAlignment: "center",
  verticalAlignment: "center", borders: border, rowHeight: 24,
};
const weekdays = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
detail.getRange(`A${firstDataRow}:L${lastDataRow}`).values = monthRows.map((row) => {
  const date = new Date(`${row.date}T00:00:00+08:00`);
  return [
    date, weekdays[date.getDay()], row.phase, Number(row.cash), Number(row.equity), Number(row.daily_pnl),
    Number(row.daily_return), Number(row.total_return), Number(row.open_positions || 0),
    Number(row.opened_positions || 0), Number(row.closed_trades || 0), row.note || "",
  ];
});
detail.getRange(`A${firstDataRow}:L${lastDataRow}`).format = {
  font: { color: colors.text, size: 10 }, borders: border, verticalAlignment: "center", rowHeight: 21,
};
detail.getRange(`A${firstDataRow}:A${lastDataRow}`).format.numberFormat = "yyyy-mm-dd";
detail.getRange(`D${firstDataRow}:F${lastDataRow}`).format.numberFormat = "#,##0.00;[Green]-#,##0.00";
detail.getRange(`G${firstDataRow}:H${lastDataRow}`).format.numberFormat = "0.00%;[Green]-0.00%";
detail.getRange(`I${firstDataRow}:K${lastDataRow}`).format.numberFormat = "0";
detail.getRange(`A${firstDataRow}:K${lastDataRow}`).format.horizontalAlignment = "center";
for (const range of [`F${firstDataRow}:H${lastDataRow}`]) {
  detail.getRange(range).conditionalFormats.add("cellIs", {
    operator: "greaterThan", formula: 0,
    format: { font: { color: colors.profit, bold: true }, fill: colors.profitFill },
  });
  detail.getRange(range).conditionalFormats.add("cellIs", {
    operator: "lessThan", formula: 0,
    format: { font: { color: colors.loss, bold: true }, fill: colors.lossFill },
  });
}
detail.mergeCells(`A${lastDataRow + 2}:L${lastDataRow + 2}`);
detail.getRange(`A${lastDataRow + 2}`).values = [[`数据来源：${options.source} ｜ 生成月份：${options.month}`]];
detail.getRange(`A${lastDataRow + 2}:L${lastDataRow + 2}`).format = {
  fill: colors.neutral, font: { color: colors.muted, size: 9 },
};
[13, 9, 13, 14, 14, 14, 13, 13, 10, 10, 12, 25].forEach((width, index) => {
  detail.getRange(`${excelColumn(index)}:${excelColumn(index)}`).format.columnWidth = width;
});
detail.freezePanes.freezeRows(6);

calendar.mergeCells("A1:G1");
calendar.getRange("A1").values = [[`${year}年${monthNumber}月模拟盘收益日历`]];
calendar.getRange("A1:G1").format = {
  fill: colors.navy, font: { bold: true, color: colors.white, size: 19 },
  horizontalAlignment: "center", verticalAlignment: "center", rowHeight: 32,
};
calendar.mergeCells("A2:G2");
calendar.getRange("A2").values = [["红色为盈利，绿色为亏损；周末与休市日不计入收益。"]];
calendar.getRange("A2:G2").format = {
  font: { color: colors.muted, italic: true, size: 10 }, horizontalAlignment: "center", rowHeight: 19,
};
const kpis = [
  ["A3:B3", "A4:B4", "月初权益（元）", "='每日明细'!B3", "#,##0.00"],
  ["C3:D3", "C4:D4", "月末权益（元）", "='每日明细'!D3", "#,##0.00"],
  ["E3:F3", "E4:F4", "月度净收益（元）", "='每日明细'!F3", "#,##0.00;[Green]-#,##0.00"],
  ["G3:G3", "G4:G4", "月度收益率", "='每日明细'!H3", "0.00%;[Green]-0.00%"],
];
for (const [labelRange, valueRange, label, formula, format] of kpis) {
  calendar.mergeCells(labelRange);
  calendar.mergeCells(valueRange);
  calendar.getRange(labelRange.split(":")[0]).values = [[label]];
  calendar.getRange(valueRange.split(":")[0]).formulas = [[formula]];
  calendar.getRange(labelRange).format = {
    fill: colors.lightBlue, font: { bold: true, color: colors.text }, horizontalAlignment: "center",
    verticalAlignment: "center", borders: border, rowHeight: 20,
  };
  calendar.getRange(valueRange).format = {
    fill: colors.white, font: { bold: true, color: colors.navy, size: 14 }, horizontalAlignment: "center",
    verticalAlignment: "center", borders: border, rowHeight: 27,
  };
  calendar.getRange(valueRange).format.numberFormat = format;
}
calendar.getRange("A6:G6").values = [["周一", "周二", "周三", "周四", "周五", "周六", "周日"]];
calendar.getRange("A6:G6").format = {
  fill: colors.blue, font: { bold: true, color: colors.white }, horizontalAlignment: "center",
  verticalAlignment: "center", borders: border, rowHeight: 22,
};

const firstDay = new Date(`${monthStart}T00:00:00+08:00`);
const mondayOffset = (firstDay.getDay() + 6) % 7;
const weekCount = Math.ceil((mondayOffset + daysInMonth) / 7);
const dateRows = Array.from({ length: weekCount }, (_, index) => 7 + index * 4);
for (let day = 1; day <= daysInMonth; day += 1) {
  const index = mondayOffset + day - 1;
  const week = Math.floor(index / 7);
  const weekday = index % 7;
  const col = excelColumn(weekday);
  const dateRow = dateRows[week];
  const dateCell = `${col}${dateRow}`;
  const date = new Date(`${options.month}-${String(day).padStart(2, "0")}T00:00:00+08:00`);
  calendar.getRange(dateCell).values = [[date]];
  calendar.getRange(`${col}${dateRow + 1}`).formulas = [[
    `=IF(COUNTIF('每日明细'!$A$${firstDataRow}:$A$${lastDataRow},${dateCell})=0,\"\",SUMIF('每日明细'!$A$${firstDataRow}:$A$${lastDataRow},${dateCell},'每日明细'!$F$${firstDataRow}:$F$${lastDataRow}))`,
  ]];
  calendar.getRange(`${col}${dateRow + 2}`).formulas = [[
    `=IF(COUNTIF('每日明细'!$A$${firstDataRow}:$A$${lastDataRow},${dateCell})=0,\"\",SUMIF('每日明细'!$A$${firstDataRow}:$A$${lastDataRow},${dateCell},'每日明细'!$G$${firstDataRow}:$G$${lastDataRow}))`,
  ]];
}
for (const dateRow of dateRows) {
  calendar.getRange(`A${dateRow}:G${dateRow}`).format = {
    font: { bold: true, color: colors.navy, size: 11 }, horizontalAlignment: "left",
    verticalAlignment: "center", borders: { top: border.top, left: border.left, right: border.right }, rowHeight: 21,
  };
  calendar.getRange(`A${dateRow}:G${dateRow}`).format.numberFormat = "m\"月\"d\"日\"";
  calendar.getRange(`A${dateRow + 1}:G${dateRow + 1}`).format = {
    font: { bold: true, color: colors.text, size: 12 }, horizontalAlignment: "center",
    verticalAlignment: "center", borders: { left: border.left, right: border.right }, rowHeight: 25,
  };
  calendar.getRange(`A${dateRow + 1}:G${dateRow + 1}`).format.numberFormat = "#,##0.00;[Green]-#,##0.00";
  calendar.getRange(`A${dateRow + 2}:G${dateRow + 2}`).format = {
    font: { color: colors.muted, size: 10 }, horizontalAlignment: "center", verticalAlignment: "center",
    borders: { bottom: border.bottom, left: border.left, right: border.right }, rowHeight: 21,
  };
  calendar.getRange(`A${dateRow + 2}:G${dateRow + 2}`).format.numberFormat = "0.00%;[Green]-0.00%";
  const resultRange = `A${dateRow + 1}:G${dateRow + 2}`;
  calendar.getRange(resultRange).conditionalFormats.add("cellIs", {
    operator: "greaterThan", formula: 0,
    format: { font: { color: colors.profit, bold: true }, fill: colors.profitFill },
  });
  calendar.getRange(resultRange).conditionalFormats.add("cellIs", {
    operator: "lessThan", formula: 0,
    format: { font: { color: colors.loss, bold: true }, fill: colors.lossFill },
  });
  calendar.getRange(`F${dateRow}:G${dateRow + 2}`).format.fill = colors.weekend;
  calendar.getRange(`A${dateRow + 3}:G${dateRow + 3}`).format.rowHeight = 5;
}
const summaryRow = dateRows.at(-1) + 4;
calendar.mergeCells(`A${summaryRow}:G${summaryRow}`);
calendar.getRange(`A${summaryRow}`).formulas = [[
  `=\"${monthNumber}月共 \"&COUNTA('每日明细'!A${firstDataRow}:A${lastDataRow})&\" 个交易日：盈利 \"&'每日明细'!J3&\" 天，亏损 \"&'每日明细'!L3&\" 天。\"`,
]];
calendar.getRange(`A${summaryRow}:G${summaryRow}`).format = {
  fill: colors.neutral, font: { bold: true, color: colors.text }, horizontalAlignment: "center",
  verticalAlignment: "center", borders: border, rowHeight: 23,
};
calendar.getRange("A:G").format.columnWidth = 17;
calendar.freezePanes.freezeRows(6);

await fs.mkdir(options.outputDir, { recursive: true });
const outputPath = path.join(options.outputDir, `${options.month}每日收益日历.xlsx`);
for (const sheetName of [`${monthNumber}月日历`, "每日明细"]) {
  const image = await workbook.render({ sheetName, autoCrop: "all", scale: 1.2, format: "png" });
  if (options.preview) {
    await fs.writeFile(
      path.join(options.outputDir, `${options.month}-${sheetName}.png`),
      new Uint8Array(await image.arrayBuffer()),
    );
  }
}
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });

const wins = monthRows.filter((row) => Number(row.daily_pnl) > 0).length;
const losses = monthRows.filter((row) => Number(row.daily_pnl) < 0).length;
console.log(`已生成：${outputPath}`);
console.log(`统计月份：${options.month}，交易日：${monthRows.length}，盈利：${wins} 天，亏损：${losses} 天`);
console.log(`月度净收益：${(endEquity - startEquity).toFixed(2)} 元，月度收益率：${((endEquity / startEquity - 1) * 100).toFixed(2)}%`);
