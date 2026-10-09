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
python3 -m quant_limitup.cli make-launchd --time 14:50 --provider sina
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
python3 -m quant_limitup.cli make-launchd --time 14:30 --provider sina --paper --send-feishu --use-minute --phase sell-force --out reports/com.quant.limitup.sell-force.plist
python3 -m quant_limitup.cli make-launchd --time 14:50 --provider sina --paper --send-feishu --refresh-stock-pool --phase buy --out reports/com.quant.limitup.buy.plist
```

模拟账户文件：

- `data/paper/account.json`：当前现金和持仓。
- `data/paper/trades.csv`：每日虚拟买卖记录。
- `data/paper/daily_returns.csv`：每天收益、余额和累计收益。
- `data/paper/candidate_reviews.csv`：每次候选复盘的信号日、结果日、命中数量和命中率历史，同时用于防止重复推送。
- `reports/rank_history.csv`：按信号日永久保存的原始候选排名。

## 月度收益日历

按指定月份生成“日历视图 + 每日明细”的 Excel。每个日期取
`data/paper/daily_returns.csv` 中最后一条账户快照作为日终结果：

```bash
./scripts/generate_monthly_report --month 2026-07
```

不传 `--month` 时，默认生成上一个自然月：

```bash
./scripts/generate_monthly_report
```

文件默认写入 `reports/monthly/YYYY-MM每日收益日历.xlsx`。如需同时保存两张工作表的
PNG 预览，可增加 `--preview`；其他参数可通过 `--help` 查看。

启用 macOS 定时任务：

```bash
cp reports/com.quant.limitup.daily.plist ~/Library/LaunchAgents/com.quant.limitup.daily.plist
launchctl load ~/Library/LaunchAgents/com.quant.limitup.daily.plist
```

阶段说明：

- `10:30 sell-morning`：检查完整的 09:35–10:30 共 12 根 5 分钟 K 线。触板或达到原有回撤及亏损条件时，按 10:30 收盘报价加卖出滑点虚拟成交；不再追溯按早盘最高价成交。
- `14:30 sell-force`：仍未卖出的隔夜持仓按有效 14:30 收盘报价加卖出滑点虚拟成交。报价缺失、重复或无效时保留持仓并提示，可在当天补齐后重试；不使用更晚报价、日线收盘价或当天最高价替代。
- `14:50 buy`：刷新股票池，先用带日期和时间校验的新浪实时快照补齐今日行情；新浪未达标时尝试整份腾讯实时快照。只有某个来源的日期、覆盖率和前收盘价连续性均通过才训练模型、生成今日候选并虚拟买入，随后单独推送买入通知。两家都未达标时暂停买入并通知；今日未完成的结果不用于训练。该定时任务使用日线模型；14:55 才完整的分钟尾盘特征此时不可用。

`buy` 阶段的弱市保护不再使用固定涨停家数。当前会综合判断：股票池涨停率低于
1.5%、市场平均涨幅不高于 -0.5%、炸板率不低于 45%；至少同时出现两项才限制为
最多买入 1 只、单只仓位不超过 20%。如果市场平均跌幅达到 -2%，直接进入弱市保护。
行情覆盖少于 1000 只时，除极端下跌外不计算弱市限仓；是否允许新买入还需先通过下面的行情完整性校验。

`daily --paper` 和 `paper-daily` 的 `buy/full` 阶段，使用 `--symbols` 指定的股票池
（默认 `config/stock_pool.csv`）校验当天行情。排除 ST、退市、北交所等不可交易标的后，
总体覆盖率须达到 90%，池中各交易所覆盖率须达到 85%；过期或未来日期的数据也会被拦截。
校验失败时暂停新增买入，不训练模型或生成新候选，原因写入 `reports/data_quality.json`
和 `reports/paper_daily_summary.json`。定时卖出阶段独立运行。
启用 `--send-feishu` / `--send-email` 时，被拦截的买入阶段也会发送“暂停买入”通知，
列出各行情源的日期、覆盖率和拦截原因；通知发送结果写入 `paper_daily_summary.json` 的
`notification_status`。通知失败不会解除买入限制，也不影响另一个已启用的通知渠道。
`full` 阶段被拦截时，仍可用未发布的单一来源日线结算已有持仓，但必须具备所有可卖持仓的
有效当日报价和前收盘价；缺少任何报价时保留账户并记录跳过原因，以便当天补齐后重试。

14:50 买入阶段先用带日期和时间的新鲜新浪实时报价，未达标再用独立的腾讯实时报价；
两者均与已接受历史的前收盘价核对，且不会拼接不同来源的股票快照。
校验均不通过即暂停新增买入。收盘后补录使用新浪收盘报价。旧的完整日线流程先尝试
腾讯日线，再尝试新浪日线；只有校验通过的整份数据才替换日线文件，不拼接两种复权口径。
训练和参数选择也会排除行情覆盖不足的信号日及其标签结束日。
历史覆盖率目前按指定股票池计算，尚未接入逐日历史股票池，不能消除幸存者偏差。

`sell-morning` 会先执行独立的候选日线复盘：只使用已完成交易日的日线最高价，按原始候选排名计算 Top10 命中率。复盘不训练模型、不修改账户或收益；休市日完成复盘后，买卖流程仍会因缺少当天分钟行情而跳过。

当前免费源实时性有限，`14:30/14:50` 的执行依赖接口当时可返回的数据；14:50 是任务启动时间，通知到达时间还取决于数据抓取和模型计算。即使配置了双源备份，两家同时缺数时仍会暂停买入。

每天 16:30 的 `refresh-close` 任务从新浪收盘报价补齐或替换当天日线，校验日期、沪深覆盖率和与历史前收盘价的连续性后才发布到 `data/raw/daily_prices.csv`。该任务只更新行情，不执行买卖；可用 `python3 -m quant_limitup.cli refresh-close` 手动运行。

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

尾盘特征统一截止到 14:55；`_1457` 列名仅为兼容旧文件保留。默认要求当天 09:35–11:30、
13:05–14:55 共 47 根完整的 5 分钟 K 线，缺失、重复、异常价格或错位时间戳均使该日分钟特征不可用。
分钟数据自动积累到 `data/raw/minute_bars.history.csv`，重复抓取不会重复累计成交量；
同次抓取出现的重复记录，即使快照换日也会保留异常标记，直到新的干净数据修正。

至少有 20 个日期、1000 条带标签的完整分钟样本才启用分钟特征，否则训练日线模型。
已启用分钟特征的模型遇到缺失分钟数据时，将其标准化后的贡献置零，避免把缺失值当作实际零收益。
`daily --paper --use-minute` 的买入阶段若分钟抓取失败或快照过期，会回退到日线模型，
日线覆盖校验仍需通过；卖出阶段保留当天分钟数据要求。

自动阈值和仓位校准使用最近最多 60 个交易日的滚动样本外预测，每 5 日一组，模型仅使用该组之前
已结束的标签训练。最后约三分之一（至少 10 日）留作独立评估，不参与选参；选择区间与留出区间之间
也排除标签跨界样本。阈值候选来自每日前排分数的分位数，并包含“不买入”方案；样本不足时明确保留默认参数。
结果记录在 `reports/strategy_optimization.json`，包括选择期、留出期、默认参数对照和逐组训练范围。
参数冻结 5 个新增交易日；正常滑动窗口更新可以复用，配置变化或稳定历史区间的数据修订会重新校准。
缓存指纹按 8 位小数规范化浮点数，避免特征重算产生的浮点误差触发重新选参；模型输入本身不做该量化。
自动选参使用分钟历史重放 10:30 / 14:30 卖出，选参期或留出期的必要执行报价不完整时保留默认参数，
并在报告中说明 `missing_execution_history`、`insufficient_execution_dates` 或
`incomplete_selection_execution` / `incomplete_holdout_execution`。分钟历史不足时不会改用日线近似收益选参。
旧版缓存会失效，历史分钟报价修订也会触发校准检查。

## 策略假设

主回测与自动选参的成交逻辑：

- 在信号日收盘附近买入。
- 有分钟历史时，复用模拟盘的 10:30 判断和 14:30 强制卖出规则，使用对应时间的报价。
- 按实际日期依次处理卖出和买入；尚未卖出的持仓继续占用资金，不提前使用未来卖出所得。
- 计入手续费、卖出印花税和滑点。
- 复用模拟盘的行情覆盖拦截、弱市限仓、最低分数、单只仓位和整手取整规则。
- 没有分钟历史时，独立日线回测仅按下一行情日收盘价结算，并明确标记 `daily_close_proxy`；不假设盘中触板价能成交。这种结果不用于自动选参。
- 有分钟历史但缺少必要报价时不补造成交；报告保留未平仓数量及 `execution_complete=false`。
- 默认入场阈值为样例数据可运行设置，真实数据必须用样本外回测重新校准。

这些假设都在 `quant_limitup/config.py` 里，可以按你的交易习惯调整。
这里的分钟收盘报价仍是模拟成交近似，不包含委托排队和成交量冲击。独立 `walk-forward` / 模型比较入口的评估规则尚未全部统一，不能与上述执行回放结果混用。

独立回测需显式指定匹配的分钟快照（会同时读取其历史归档），避免样例特征误用真实行情：

```bash
python3 -m quant_limitup.cli backtest --minute-bars-file data/raw/minute_bars.csv
```

不指定该参数时使用标注清楚的日线收盘价近似；指定了但无有效分钟数据时，报告缺报价和未平仓情况。

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
