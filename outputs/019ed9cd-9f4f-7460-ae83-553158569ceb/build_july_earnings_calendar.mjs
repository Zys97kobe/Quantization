import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "/Users/baibaibaibaibai/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const projectRoot = "/Users/baibaibaibaibai/git_repo/git_hub/Quantization";
const sourcePath = path.join(projectRoot, "data/paper/daily_returns.csv");
const outputDir = path.join(projectRoot, "outputs/019ed9cd-9f4f-7460-ae83-553158569ceb");
const outputPath = path.join(outputDir, "2026年7月每日收益日历.xlsx");

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
  const headers = rows[0];
  return rows.slice(1).filter((r) => r.some((v) => v !== "")).map((r) =>
    Object.fromEntries(headers.map((h, i) => [h, r[i] ?? ""]))
  );
}

const rawRows = parseCsv(await fs.readFile(sourcePath, "utf8"));
const julyLastByDate = new Map();
for (const row of rawRows) {
  if (row.date.startsWith("2026-07")) julyLastByDate.set(row.date, row);
}
const julyRows = [...julyLastByDate.values()];
const priorRow = rawRows.filter((row) => row.date < "2026-07-01").at(-1);

if (!priorRow || julyRows.length === 0) {
  throw new Error("未找到 2026 年 7 月收益记录或月初基准记录");
}

const workbook = Workbook.create();
const calendar = workbook.worksheets.add("7月日历");
const detail = workbook.worksheets.add("每日明细");

const colors = {
  navy: "#17365D",
  blue: "#2F75B5",
  lightBlue: "#D9EAF7",
  paleBlue: "#EEF5FB",
  white: "#FFFFFF",
  text: "#243447",
  muted: "#687386",
  border: "#C9D4E2",
  weekend: "#F1F3F5",
  profit: "#C00000",
  profitFill: "#FCE8E6",
  loss: "#008000",
  lossFill: "#E8F5E9",
  neutral: "#F8FAFC",
};

const thinBorder = {
  top: { style: "thin", color: colors.border },
  bottom: { style: "thin", color: colors.border },
  left: { style: "thin", color: colors.border },
  right: { style: "thin", color: colors.border },
};

// 每日明细：导入该日最后一条账户快照，保留阶段字段便于审计。
detail.mergeCells("A1:L1");
detail.getRange("A1").values = [["2026年7月模拟盘每日收益明细"]];
detail.getRange("A1:L1").format = {
  fill: colors.navy,
  font: { bold: true, color: colors.white, size: 18 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  rowHeight: 30,
};

detail.getRange("A3:L3").format = {
  fill: colors.paleBlue,
  font: { bold: true, color: colors.text },
  borders: thinBorder,
  verticalAlignment: "center",
  rowHeight: 23,
};
detail.getRange("A3:L3").values = [[
  "月初权益（元）", Number(priorRow.equity),
  "月末权益（元）", null,
  "月度净收益（元）", null,
  "月度收益率", null,
  "盈利日", null,
  "亏损日", null,
]];

const firstDataRow = 7;
const lastDataRow = firstDataRow + julyRows.length - 1;
detail.getRange("D3").formulas = [[`=E${lastDataRow}`]];
detail.getRange("F3").formulas = [["=D3-B3"]];
detail.getRange("H3").formulas = [["=D3/B3-1"]];
detail.getRange("J3").formulas = [[`=COUNTIF(F${firstDataRow}:F${lastDataRow},\">0\")`]];
detail.getRange("L3").formulas = [[`=COUNTIF(F${firstDataRow}:F${lastDataRow},\"<0\")`]];
for (const cell of ["B3", "D3", "F3"]) detail.getRange(cell).format.numberFormat = "#,##0.00;[Green]-#,##0.00";
detail.getRange("H3").format.numberFormat = "0.00%;[Green]-0.00%";

detail.getRange("A5:L5").merge();
detail.getRange("A5").values = [["口径：每个自然日取 daily_returns.csv 中最后一条账户快照；当日盈亏相对上一交易日最终权益。"]];
detail.getRange("A5:L5").format = {
  fill: colors.neutral,
  font: { italic: true, color: colors.muted, size: 10 },
  verticalAlignment: "center",
  rowHeight: 20,
};

const headers = [["日期", "星期", "日终阶段", "现金（元）", "日终权益（元）", "当日盈亏（元）", "当日收益率", "累计收益率", "当前持仓", "新开仓", "已结算交易", "备注"]];
detail.getRange("A6:L6").values = headers;
detail.getRange("A6:L6").format = {
  fill: colors.blue,
  font: { bold: true, color: colors.white },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: thinBorder,
  rowHeight: 24,
};

const weekdays = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
const detailValues = julyRows.map((row) => {
  const date = new Date(`${row.date}T00:00:00+08:00`);
  const note = row.date === "2026-07-14" ? "最终快照为 sell-force" : (row.note || "");
  return [
    date,
    weekdays[date.getDay()],
    row.phase,
    Number(row.cash),
    Number(row.equity),
    Number(row.daily_pnl),
    Number(row.daily_return),
    Number(row.total_return),
    Number(row.open_positions),
    Number(row.opened_positions),
    Number(row.closed_trades),
    note,
  ];
});
detail.getRange(`A${firstDataRow}:L${lastDataRow}`).values = detailValues;
detail.getRange(`A${firstDataRow}:L${lastDataRow}`).format = {
  font: { color: colors.text, size: 10 },
  borders: thinBorder,
  verticalAlignment: "center",
  rowHeight: 21,
};
detail.getRange(`A${firstDataRow}:A${lastDataRow}`).format.numberFormat = "yyyy-mm-dd";
detail.getRange(`D${firstDataRow}:F${lastDataRow}`).format.numberFormat = "#,##0.00;[Green]-#,##0.00";
detail.getRange(`G${firstDataRow}:H${lastDataRow}`).format.numberFormat = "0.00%;[Green]-0.00%";
detail.getRange(`I${firstDataRow}:K${lastDataRow}`).format.numberFormat = "0";
detail.getRange(`A${firstDataRow}:K${lastDataRow}`).format.horizontalAlignment = "center";
detail.getRange(`F${firstDataRow}:H${lastDataRow}`).conditionalFormats.add("cellIs", {
  operator: "greaterThan",
  formula: 0,
  format: { font: { color: colors.profit, bold: true }, fill: colors.profitFill },
});
detail.getRange(`F${firstDataRow}:H${lastDataRow}`).conditionalFormats.add("cellIs", {
  operator: "lessThan",
  formula: 0,
  format: { font: { color: colors.loss, bold: true }, fill: colors.lossFill },
});

detail.getRange(`A${lastDataRow + 2}:L${lastDataRow + 2}`).merge();
detail.getRange(`A${lastDataRow + 2}`).values = [[`数据来源：${sourcePath} ｜ 生成日期：2026-08-05`]];
detail.getRange(`A${lastDataRow + 2}:L${lastDataRow + 2}`).format = {
  font: { color: colors.muted, size: 9 },
  fill: colors.neutral,
};

const widths = [13, 9, 13, 14, 14, 14, 13, 13, 10, 10, 12, 25];
widths.forEach((width, i) => {
  const col = String.fromCharCode(65 + i);
  detail.getRange(`${col}:${col}`).format.columnWidth = width;
});
detail.freezePanes.freezeRows(6);

// 日历视图：每周三行（日期、当日盈亏、收益率），休市日仅显示日期。
calendar.mergeCells("A1:G1");
calendar.getRange("A1").values = [["2026年7月模拟盘收益日历"]];
calendar.getRange("A1:G1").format = {
  fill: colors.navy,
  font: { bold: true, color: colors.white, size: 19 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  rowHeight: 32,
};
calendar.mergeCells("A2:G2");
calendar.getRange("A2").values = [["红色为盈利，绿色为亏损；周末与休市日不计入收益。"]];
calendar.getRange("A2:G2").format = {
  font: { color: colors.muted, italic: true, size: 10 },
  horizontalAlignment: "center",
  rowHeight: 19,
};

const kpis = [
  ["A3:B3", "A4:B4", "月初权益（元）", "='每日明细'!B3", "#,##0.00"],
  ["C3:D3", "C4:D4", "月末权益（元）", "='每日明细'!D3", "#,##0.00"],
  ["E3:F3", "E4:F4", "月度净收益（元）", "='每日明细'!F3", "#,##0.00;[Green]-#,##0.00"],
  ["G3:G3", "G4:G4", "月度收益率", "='每日明细'!H3", "0.00%;[Green]-0.00%"],
];
for (const [labelRange, valueRange, label, formula, numberFormat] of kpis) {
  if (labelRange.includes(":")) calendar.mergeCells(labelRange);
  if (valueRange.includes(":")) calendar.mergeCells(valueRange);
  calendar.getRange(labelRange.split(":")[0]).values = [[label]];
  calendar.getRange(valueRange.split(":")[0]).formulas = [[formula]];
  calendar.getRange(labelRange).format = {
    fill: colors.lightBlue,
    font: { bold: true, color: colors.text },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: thinBorder,
    rowHeight: 20,
  };
  calendar.getRange(valueRange).format = {
    fill: colors.white,
    font: { bold: true, color: colors.navy, size: 14 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: thinBorder,
    rowHeight: 27,
  };
  calendar.getRange(valueRange).format.numberFormat = numberFormat;
}

calendar.getRange("A6:G6").values = [["周一", "周二", "周三", "周四", "周五", "周六", "周日"]];
calendar.getRange("A6:G6").format = {
  fill: colors.blue,
  font: { bold: true, color: colors.white },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: thinBorder,
  rowHeight: 22,
};

const dateRows = [7, 11, 15, 19, 23];
const firstOfMonth = new Date("2026-07-01T00:00:00+08:00");
const mondayIndex = (firstOfMonth.getDay() + 6) % 7;
for (let day = 1; day <= 31; day += 1) {
  const index = mondayIndex + day - 1;
  const week = Math.floor(index / 7);
  const weekday = index % 7;
  const col = String.fromCharCode(65 + weekday);
  const dateRow = dateRows[week];
  const pnlRow = dateRow + 1;
  const returnRow = dateRow + 2;
  const date = new Date(`2026-07-${String(day).padStart(2, "0")}T00:00:00+08:00`);
  const dateCell = `${col}${dateRow}`;
  calendar.getRange(dateCell).values = [[date]];
  calendar.getRange(`${col}${pnlRow}`).formulas = [[
    `=IF(COUNTIF('每日明细'!$A$${firstDataRow}:$A$${lastDataRow},${dateCell})=0,\"\",SUMIF('每日明细'!$A$${firstDataRow}:$A$${lastDataRow},${dateCell},'每日明细'!$F$${firstDataRow}:$F$${lastDataRow}))`,
  ]];
  calendar.getRange(`${col}${returnRow}`).formulas = [[
    `=IF(COUNTIF('每日明细'!$A$${firstDataRow}:$A$${lastDataRow},${dateCell})=0,\"\",SUMIF('每日明细'!$A$${firstDataRow}:$A$${lastDataRow},${dateCell},'每日明细'!$G$${firstDataRow}:$G$${lastDataRow}))`,
  ]];
}

for (const dateRow of dateRows) {
  const pnlRow = dateRow + 1;
  const returnRow = dateRow + 2;
  calendar.getRange(`A${dateRow}:G${dateRow}`).format = {
    font: { bold: true, color: colors.navy, size: 11 },
    horizontalAlignment: "left",
    verticalAlignment: "center",
    borders: { top: thinBorder.top, left: thinBorder.left, right: thinBorder.right },
    rowHeight: 21,
  };
  calendar.getRange(`A${dateRow}:G${dateRow}`).format.numberFormat = "m\"月\"d\"日\"";
  calendar.getRange(`A${pnlRow}:G${pnlRow}`).format = {
    font: { bold: true, color: colors.text, size: 12 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { left: thinBorder.left, right: thinBorder.right },
    rowHeight: 25,
  };
  calendar.getRange(`A${pnlRow}:G${pnlRow}`).format.numberFormat = "#,##0.00;[Green]-#,##0.00";
  calendar.getRange(`A${returnRow}:G${returnRow}`).format = {
    font: { color: colors.muted, size: 10 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    borders: { bottom: thinBorder.bottom, left: thinBorder.left, right: thinBorder.right },
    rowHeight: 21,
  };
  calendar.getRange(`A${returnRow}:G${returnRow}`).format.numberFormat = "0.00%;[Green]-0.00%";
  calendar.getRange(`A${pnlRow}:G${returnRow}`).conditionalFormats.add("cellIs", {
    operator: "greaterThan",
    formula: 0,
    format: { font: { color: colors.profit, bold: true }, fill: colors.profitFill },
  });
  calendar.getRange(`A${pnlRow}:G${returnRow}`).conditionalFormats.add("cellIs", {
    operator: "lessThan",
    formula: 0,
    format: { font: { color: colors.loss, bold: true }, fill: colors.lossFill },
  });
  calendar.getRange(`F${dateRow}:G${returnRow}`).format.fill = colors.weekend;
}

for (const spacer of [10, 14, 18, 22]) {
  calendar.getRange(`A${spacer}:G${spacer}`).format.rowHeight = 5;
}
calendar.mergeCells("A27:G27");
calendar.getRange("A27").formulas = [[`=\"7月共 \"&COUNTA('每日明细'!A${firstDataRow}:A${lastDataRow})&\" 个交易日：盈利 \"&'每日明细'!J3&\" 天，亏损 \"&'每日明细'!L3&\" 天。\"`]];
calendar.getRange("A27:G27").format = {
  fill: colors.neutral,
  font: { bold: true, color: colors.text },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: thinBorder,
  rowHeight: 23,
};
calendar.getRange("A:G").format.columnWidth = 17;
calendar.freezePanes.freezeRows(6);

await fs.mkdir(outputDir, { recursive: true });
for (const sheetName of ["7月日历", "每日明细"]) {
  const image = await workbook.render({ sheetName, autoCrop: "all", scale: 1.2, format: "png" });
  await fs.writeFile(path.join(outputDir, `${sheetName}.png`), new Uint8Array(await image.arrayBuffer()));
}
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);

console.log(JSON.stringify({ outputPath, sourcePath, julyRows: julyRows.length, firstDataRow, lastDataRow }, null, 2));
