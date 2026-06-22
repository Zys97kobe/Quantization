# A 股涨停预测模拟系统 MVP

这个项目先解决一件事：每天尾盘前，用历史真实行情或样例行情计算候选股票的次日涨停概率，并在模拟盘里验证买入、卖出、仓位和回撤。

当前版本不连接券商、不自动实盘下单。它是研究和模拟交易框架，后续可以接入 AkShare、TuShare、Wind、聚宽、券商柜台或你已有的数据源。

## 快速开始

```bash
python3 -m quant_limitup.cli init-sample
python3 -m quant_limitup.cli run-pipeline
```

运行后会生成：

- `data/raw/daily_prices.csv`：样例日线行情。
- `data/processed/features.csv`：训练特征和标签。
- `models/limitup_logistic.json`：涨停概率基线模型。
- `reports/latest_rank.csv`：最新交易日候选股票排名。
- `reports/backtest_trades.csv`：模拟交易明细。
- `reports/backtest_summary.json`：回测指标。
- `reports/dashboard.html`：可直接打开的本地报告。

## 接入真实数据

准备一个 CSV，至少包含以下字段：

```text
date,symbol,name,board,is_st,open,high,low,close,volume,amount,turnover,free_float_mkt_cap
```

说明：

- `date`：交易日期，格式 `YYYY-MM-DD`。
- `symbol`：证券代码，例如 `600000.SH`、`000001.SZ`。
- `board`：`main`、`star`、`chinext`、`bse` 之一。
- `is_st`：`0/1`。
- `turnover`：换手率，按小数写，例如 `0.082`。
- `free_float_mkt_cap`：流通市值，单位元。

然后运行：

```bash
python3 -m quant_limitup.cli build-dataset --prices path/to/your_daily_prices.csv
python3 -m quant_limitup.cli train
python3 -m quant_limitup.cli rank
python3 -m quant_limitup.cli backtest
```

也可以用 AkShare 自动抓取东方财富 A 股日线数据：

```bash
python3 -m pip install akshare
python3 -m quant_limitup.cli fetch-akshare --start-date 20240101
python3 -m quant_limitup.cli run-real
```

`fetch-akshare` 会写入 `data/raw/daily_prices.csv`，后续流程和 CSV 数据完全一致。免费数据源可能限流或字段变化，第一次建议先用较短日期验证，例如最近 60-120 个交易日。

免 token 备用源是新浪日线 K 线，默认用 `config/stock_pool.csv` 股票池：

```bash
python3 -m quant_limitup.cli update-stock-pool
python3 -m quant_limitup.cli fetch-sina --days 260
python3 -m quant_limitup.cli run-real
```

更完整的方案是 TuShare Pro，需要先配置有 `daily`、`daily_basic`、`stock_basic` 权限的 token：

```bash
python3 -m pip install tushare
mkdir -p config
printf "你的 token" > config/tushare_token.txt
python3 -m quant_limitup.cli fetch-tushare --start-date 20240101
python3 -m quant_limitup.cli run-real
```

TuShare 的 `daily`、`daily_basic`、`stock_basic` 权限和调用频率取决于你的账号积分。生产使用建议优先用 TuShare、Wind、聚宽或券商数据；新浪源可以先跑通自动化，但缺少换手率和流通市值等字段。

## 每天自动运行

先确认手动命令可跑通：

```bash
python3 -m quant_limitup.cli daily --provider sina
```

生成 macOS `launchd` 定时任务模板：

```bash
python3 -m quant_limitup.cli make-launchd --time 15:10 --provider sina
```

它会生成 `reports/com.quant.limitup.daily.plist`。你可以把里面的 `ProgramArguments` 和 `StartCalendarInterval` 检查一遍，再复制到 `~/Library/LaunchAgents/` 并加载。建议先只做“生成候选清单 + 模拟盘记录”，不要自动真实下单。

## 每日虚拟交易和通知

初始化 10000 元模拟账户：

```bash
python3 -m quant_limitup.cli reset-paper --initial-cash 10000
```

飞书机器人通知：

```bash
printf "你的飞书机器人 webhook" > config/feishu_webhook.txt
python3 -m quant_limitup.cli paper-daily --send-feishu
```

每日自动运行、虚拟交易并推送飞书。当前推荐拆成三个阶段：

```bash
python3 -m quant_limitup.cli update-stock-pool
python3 -m quant_limitup.cli make-launchd --time 10:30 --provider sina --paper --send-feishu --use-minute --phase sell-morning --out reports/com.quant.limitup.sell-morning.plist
python3 -m quant_limitup.cli make-launchd --time 14:50 --provider sina --paper --send-feishu --use-minute --phase sell-force --out reports/com.quant.limitup.sell-force.plist
python3 -m quant_limitup.cli make-launchd --time 15:10 --provider sina --paper --send-feishu --refresh-stock-pool --use-minute --minute-mode top --minute-top-n 300 --phase buy --out reports/com.quant.limitup.buy.plist
```

模拟账户文件：

- `data/paper/account.json`：当前现金和持仓。
- `data/paper/trades.csv`：每日虚拟买卖记录。
- `data/paper/daily_returns.csv`：每天收益、余额和累计收益。
- `data/paper/candidate_reviews.csv`：每次候选复盘的信号日、结果日、命中数量和命中率历史，同时用于防止重复推送。
- `reports/rank_history.csv`：按信号日永久保存的原始候选排名。

启用 macOS 定时任务：

```bash
cp reports/com.quant.limitup.daily.plist ~/Library/LaunchAgents/com.quant.limitup.daily.plist
launchctl load ~/Library/LaunchAgents/com.quant.limitup.daily.plist
```

阶段说明：

- `10:30 sell-morning`：只处理已有持仓；如果 09:30-10:30 分钟线触及涨停，或冲高回落超过 3%，虚拟卖出并单独推送卖出通知。
- `14:50 sell-force`：仍未卖出的隔夜持仓强制虚拟卖出，并单独推送卖出通知。
- `15:10 buy`：刷新股票池和行情，训练模型，生成今日候选并虚拟买入，单独推送买入通知。

`sell-morning` 会先执行独立的候选日线复盘：只使用已完成交易日的日线最高价，按原始候选排名计算 Top10 命中率。复盘不训练模型、不修改账户或收益；休市日完成复盘后，买卖流程仍会因缺少当天分钟行情而跳过。

当前免费源实时性有限，`14:50/15:10` 的执行依赖新浪接口当时可返回的数据；专业实盘前应替换为稳定实时行情源。

启用 `--use-minute` 后，系统会额外抓取新浪免费分钟 K 线，并生成尾盘聚合特征：

- `tail_ret_1430_1457`
- `tail_ret_1450_1457`
- `tail_volume_ratio`
- `tail_amount_ratio`
- `tail_volume_vs_5d`
- `tail_high_break`
- `tail_close_to_high`
- `tail_limit_gap`
- `tail_vwap_deviation`
- `tail_pullback`
- `tail_range`

免费分钟线历史较短且接口稳定性一般；抓取失败时应检查 `reports/daily.err.log` 和 `data/raw/minute_bars.failures.csv`。

## 策略假设

默认回测逻辑：

- 在信号日收盘附近买入。
- 次日如果盘中触及涨停价，按涨停价附近卖出。
- 如果未触及涨停，次日收盘卖出。
- 计入手续费、卖出印花税和滑点。
- 按概率排序，每天最多买入固定数量股票。
- 默认入场阈值为样例数据可运行设置，真实数据必须用样本外回测重新校准。

这些假设都在 `quant_limitup/config.py` 里，可以按你的交易习惯调整。

## 后续增强路线

1. 接入真实分钟线和盘口数据，强化尾盘特征。
2. 用 LightGBM/XGBoost 替换基线逻辑回归。
3. 加入题材、公告、龙虎榜、融资融券等事件数据。
4. 加入实时 14:30-14:55 扫描任务。
5. 模拟盘稳定后，再做券商交易接口，但必须保留人工确认和风控开关。

## 样本外评估

用过去数据滚动训练、预测下一天，评估模型是否真的有排序能力：

```bash
python3 -m quant_limitup.cli walk-forward --train-days 30 --min-train-rows 200 --epochs 300 --pretrain-final
```

输出：

- `reports/walk_forward_summary.json`
- `reports/walk_forward_predictions.csv`
- `reports/walk_forward_trades.csv`

重点看 `top3_hit_rate`、`top5_days_with_hit`、`trade_limit_hit_rate`、`total_return` 和 `max_drawdown`。不要用普通准确率判断涨停模型，因为正例极少。
