import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const payloadPath = process.argv[2];
const outputPath = process.argv[3];
if (!payloadPath || !outputPath) {
  throw new Error("Usage: node build_v2_diagnostics_workbook.mjs <payload.json> <output.xlsx>");
}
const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
const outputDir = path.dirname(outputPath);
await fs.mkdir(outputDir, { recursive: true });
const previewDir = path.join(outputDir, "workbook_previews");
await fs.mkdir(previewDir, { recursive: true });

const wb = Workbook.create();
const colors = {
  navy: "#17324D",
  blue: "#2E86AB",
  teal: "#3D8D7A",
  lightBlue: "#EAF3F8",
  lightTeal: "#EAF5F1",
  amber: "#D99B2B",
  red: "#C95C54",
  lightRed: "#FBEDEC",
  gray: "#F3F5F7",
  border: "#D5DCE3",
  text: "#1F2933",
  white: "#FFFFFF",
};

function applyTitle(sheet, range, text) {
  sheet.mergeCells(range);
  const cell = sheet.getRange(range.split(":")[0]);
  cell.values = [[text]];
  sheet.getRange(range).format = {
    fill: colors.navy,
    font: { bold: true, color: colors.white, size: 18 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
}

function styleHeader(range) {
  range.format = {
    fill: colors.blue,
    font: { bold: true, color: colors.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "all", style: "thin", color: colors.border },
  };
}

function writeTable(sheet, startRow, startCol, headers, records, tableName) {
  const matrix = [headers, ...records.map((record) => headers.map((header) => record[header] ?? null))];
  const range = sheet.getRangeByIndexes(startRow, startCol, matrix.length, headers.length);
  range.values = matrix;
  styleHeader(sheet.getRangeByIndexes(startRow, startCol, 1, headers.length));
  if (records.length > 0) {
    const table = sheet.tables.add(range, true, tableName);
    table.style = "TableStyleMedium2";
    table.showBandedRows = true;
  }
  return range;
}

function addKpi(sheet, range, label, value, fill) {
  sheet.mergeCells(range);
  const [from] = range.split(":");
  sheet.getRange(from).values = [[`${label}\n${value}`]];
  sheet.getRange(range).format = {
    fill,
    font: { bold: true, color: colors.text, size: 13 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "outside", style: "medium", color: colors.border },
  };
}

const dashboard = wb.worksheets.add("Dashboard");
dashboard.showGridLines = false;
applyTitle(dashboard, "A1:P2", "Stage C Recommended v2 — Engineering & Error Diagnostics");
dashboard.getRange("A3:P3").merge();
dashboard.getRange("A3").values = [["30-stock validation diagnostics · three frozen seeds · test not evaluated or used for selection"]];
dashboard.getRange("A3:P3").format = {
  fill: colors.lightBlue,
  font: { italic: true, color: colors.navy },
  horizontalAlignment: "center",
};

const eng = payload.summary.engineering;
const disagreement = payload.summary.component_disagreement;
addKpi(dashboard, "A5:D7", "Parameters", Number(eng.total_parameters).toLocaleString(), colors.lightBlue);
addKpi(dashboard, "E5:H7", "Checkpoint size", `${(eng.total_checkpoint_bytes / 1024).toFixed(1)} KB`, colors.lightTeal);
addKpi(dashboard, "I5:L7", "Median inference / 120", `${eng.mean_inference_median_seconds_120.toFixed(4)} s`, colors.lightBlue);
addKpi(dashboard, "M5:P7", "Throughput", `${eng.mean_throughput_samples_per_second.toFixed(0)} samples/s`, colors.lightTeal);
addKpi(dashboard, "A9:D11", "Component correlation", disagreement.mean_prediction_correlation.toFixed(3), colors.gray);
addKpi(dashboard, "E9:H11", "Sign disagreement", `${(disagreement.mean_sign_disagreement_rate * 100).toFixed(1)}%`, colors.lightRed);
addKpi(dashboard, "I9:L11", "Mean absolute disagreement", disagreement.mean_absolute_disagreement.toFixed(4), colors.gray);
addKpi(dashboard, "M9:P11", "Ensemble MAE gain", disagreement.mean_ensemble_gain_vs_average.toFixed(4), colors.lightTeal);

dashboard.getRange("A13:F13").merge();
dashboard.getRange("A13").values = [["Key diagnostic conclusions"]];
dashboard.getRange("A13:F13").format = { fill: colors.teal, font: { bold: true, color: colors.white } };
dashboard.getRange("A14:F19").merge();
dashboard.getRange("A14").values = [[
  "• The two components are strongly complementary: mean correlation is near zero and signs disagree in 42.2% of rows.\n" +
  "• The ensemble is most reliable for low/medium absolute returns; high-magnitude moves remain the main error source.\n" +
  "• Strong positive-return weeks are systematically underpredicted; strong declines are also shrunk toward zero.\n" +
  "• High component disagreement creates more averaging benefit, but also a larger gap versus an oracle component selector.\n" +
  "• Five stocks dominate the largest errors and should be prioritized in the next diagnostic cycle."
]];
dashboard.getRange("A14:F19").format = {
  fill: colors.gray, wrapText: true, verticalAlignment: "top",
  borders: { preset: "outside", style: "thin", color: colors.border },
};

const quintileOrder = [
  "Q1 strongest decline", "Q2 decline", "Q3 near zero", "Q4 rise", "Q5 strongest rise",
];
const quintiles = payload.return_groups
  .filter((row) => row.group_type === "return_quintile")
  .sort((a, b) => quintileOrder.indexOf(a.group) - quintileOrder.indexOf(b.group));
dashboard.getRange("A22:B27").values = [
  ["Return quintile", "MAE"],
  ...quintiles.map((row) => [row.group, row.mae]),
];
styleHeader(dashboard.getRange("A22:B22"));
dashboard.getRange("B23:B27").format.numberFormat = "0.0000";
const returnChart = dashboard.charts.add("bar", dashboard.getRange("A22:B27"));
returnChart.title = "MAE by realized-return quintile";
returnChart.hasLegend = false;
returnChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
returnChart.yAxis = { numberFormatCode: "0.000" };
returnChart.setPosition("H13", "P29");

const worstStocks = [...payload.stock_diagnostics].sort((a, b) => b.mae - a.mae).slice(0, 10);
dashboard.getRange("A31:B41").values = [
  ["Worst stock", "MAE"],
  ...worstStocks.map((row) => [row.stock_code, row.mae]),
];
styleHeader(dashboard.getRange("A31:B31"));
dashboard.getRange("B32:B41").format.numberFormat = "0.0000";
const stockChart = dashboard.charts.add("bar", dashboard.getRange("A31:B41"));
stockChart.title = "Top 10 stock-level MAE";
stockChart.hasLegend = false;
stockChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 8 } };
stockChart.yAxis = { numberFormatCode: "0.000" };
stockChart.setPosition("H31", "P49");

const disagreementOrder = ["D1 low", "D2", "D3", "D4 high"];
const dashboardDisagreement = [...payload.disagreement_buckets].sort(
  (a, b) => disagreementOrder.indexOf(a.disagreement_quartile) - disagreementOrder.indexOf(b.disagreement_quartile),
);
dashboard.getRange("A44:C48").values = [
  ["Disagreement bucket", "MAE gain", "Oracle gap"],
  ...dashboardDisagreement.map((row) => [
    row.disagreement_quartile, row.ensemble_gain_vs_average, row.ensemble_gap_vs_oracle,
  ]),
];
styleHeader(dashboard.getRange("A44:C44"));
dashboard.getRange("B45:C48").format.numberFormat = "0.0000";
const disagreementChart = dashboard.charts.add("bar", dashboard.getRange("A44:C48"));
disagreementChart.title = "Averaging gain vs oracle gap";
disagreementChart.hasLegend = true;
disagreementChart.yAxis = { numberFormatCode: "0.000" };
disagreementChart.setPosition("H51", "P66");
dashboard.getRange("A:A").format.columnWidth = 26;
dashboard.getRange("B:P").format.columnWidth = 13;
dashboard.getRange("4:66").format.rowHeight = 20;
dashboard.freezePanes.freezeRows(3);

const engineering = wb.worksheets.add("Engineering_Cost");
engineering.showGridLines = false;
applyTitle(engineering, "A1:M2", "Engineering Cost and Runtime Benchmark");
const engineeringHeaders = [
  "scope", "component", "runs", "parameters", "checkpoint_bytes", "training_seconds_mean",
  "training_seconds_std", "load_seconds_mean", "inference_median_seconds_120",
  "inference_p95_seconds_120", "throughput_samples_per_second", "rss_delta_total_bytes",
  "data_load_seconds_shared",
];
writeTable(engineering, 3, 0, engineeringHeaders, payload.engineering_summary, "EngineeringSummaryTable");
engineering.getRange("D5:E10").format.numberFormat = "#,##0";
engineering.getRange("F5:J10").format.numberFormat = "0.0000";
engineering.getRange("K5:K10").format.numberFormat = "#,##0";
engineering.getRange("L5:L10").format.numberFormat = "#,##0";
engineering.getRange("A:M").format.columnWidth = 18;
engineering.getRange("B:B").format.columnWidth = 28;
engineering.freezePanes.freezeRows(4);

const stockSheet = wb.worksheets.add("Stock_Diagnostics");
stockSheet.showGridLines = false;
applyTitle(stockSheet, "A1:R2", "Per-stock Error Diagnostics — 3 Seeds × 4 Validation Weeks");
const stockHeaders = [
  "stock_code", "stock_name", "industry", "samples", "mae", "rmse", "bias",
  "direction_accuracy", "direction_f1", "temporal_mae", "fixed_graph_mae",
  "mae_improvement_vs_temporal_pct", "mae_improvement_vs_fixed_pct",
  "mean_component_disagreement", "component_sign_disagreement_rate",
  "mean_cross_seed_prediction_std",
];
writeTable(stockSheet, 3, 0, stockHeaders, payload.stock_diagnostics, "StockDiagnosticsTable");
stockSheet.getRange("E5:K40").format.numberFormat = "0.0000";
stockSheet.getRange("L5:M40").format.numberFormat = "0.0\"%\"";
stockSheet.getRange("N5:P40").format.numberFormat = "0.0000";
stockSheet.getRange("H5:I40").format.numberFormat = "0.0%";
stockSheet.getRange("O5:O40").format.numberFormat = "0.0%";
stockSheet.getRange("E5:E40").conditionalFormats.add("colorScale", {
  colors: ["#63BE7B", "#FFEB84", "#F8696B"], thresholds: ["min", "50%", "max"],
});
stockSheet.getRange("A:R").format.columnWidth = 15;
stockSheet.getRange("B:C").format.columnWidth = 18;
stockSheet.freezePanes.freezeRows(4);

const returns = wb.worksheets.add("Return_Groups");
returns.showGridLines = false;
applyTitle(returns, "A1:N2", "Error Diagnostics by Realized Return Group");
const returnHeaders = [
  "group_type", "group", "samples", "mae", "rmse", "bias", "direction_accuracy",
  "direction_f1", "mean_target_return", "temporal_mae", "fixed_graph_mae",
  "mean_component_disagreement", "ensemble_gain_vs_average",
];
writeTable(returns, 3, 0, returnHeaders, payload.return_groups, "ReturnGroupsTable");
returns.getRange("D5:M30").format.numberFormat = "0.0000";
returns.getRange("G5:H30").format.numberFormat = "0.0%";
returns.getRange("D5:D30").conditionalFormats.add("colorScale", {
  colors: ["#63BE7B", "#FFEB84", "#F8696B"], thresholds: ["min", "50%", "max"],
});
returns.getRange("A:N").format.columnWidth = 18;
returns.getRange("B:B").format.columnWidth = 26;
returns.freezePanes.freezeRows(4);

const disagree = wb.worksheets.add("Disagreement");
disagree.showGridLines = false;
applyTitle(disagree, "A1:J2", "Component Disagreement and Ensemble Benefit");
const disagreementHeaders = [
  "disagreement_quartile", "samples", "mean_component_disagreement", "ensemble_mae",
  "temporal_mae", "fixed_graph_mae", "ensemble_gain_vs_average", "ensemble_gap_vs_oracle",
  "ensemble_beats_both_rate", "component_sign_disagreement_rate",
];
writeTable(disagree, 3, 0, disagreementHeaders, payload.disagreement_buckets, "DisagreementBucketsTable");
disagree.getRange("C5:H15").format.numberFormat = "0.0000";
disagree.getRange("I5:J15").format.numberFormat = "0.0%";
disagree.getRange("A:J").format.columnWidth = 20;
disagree.freezePanes.freezeRows(4);

const samplesSheet = wb.worksheets.add("Sample_Details");
samplesSheet.showGridLines = false;
applyTitle(samplesSheet, "A1:T2", "Sample-level Diagnostics");
const sampleHeaders = [
  "seed", "stock_code", "stock_name", "industry", "trade_date", "target_date", "target_return",
  "ensemble_prediction", "temporal_prediction", "fixed_graph_prediction", "ensemble_abs_error",
  "temporal_abs_error", "fixed_graph_abs_error", "component_disagreement",
  "component_sign_disagreement", "ensemble_gain_vs_average", "ensemble_gap_vs_oracle",
  "better_component", "return_quintile", "magnitude_group",
];
writeTable(samplesSheet, 3, 0, sampleHeaders, payload.sample_diagnostics, "SampleDiagnosticsTable");
samplesSheet.getRange("G5:Q400").format.numberFormat = "0.0000";
samplesSheet.getRange("A:T").format.columnWidth = 16;
samplesSheet.getRange("C:D").format.columnWidth = 18;
samplesSheet.getRange("R:T").format.columnWidth = 24;
samplesSheet.freezePanes.freezeRows(4);

const methods = wb.worksheets.add("Methodology");
methods.showGridLines = false;
applyTitle(methods, "A1:H2", "Methodology, Scope and Interpretation");
const methodRows = [
  ["Scope", "30 stocks, validation only, seeds 20260723/20260724/20260725."],
  ["Recommended model", "0.5 × temporal-only Transformer + 0.5 × fixed temporal graph control."],
  ["Engineering benchmark", "CPU, 10 timed end-to-end prediction repetitions after one warm-up; sequence reconstruction included."],
  ["Stock diagnostics", "12 observations per stock: 4 validation weeks × 3 seeds."],
  ["Return quintiles", "Computed from the 120 unique validation targets, then applied identically to all three seeds."],
  ["Component gain", "Average component absolute error minus ensemble absolute error; positive means averaging reduced MAE."],
  ["Oracle gap", "Ensemble absolute error minus the better component absolute error; measures remaining routing opportunity."],
  ["Evidence boundary", "Development validation diagnostics. Test is not evaluated or used for selection."],
  ["Interpretation warning", "Small per-stock samples and reused validation data do not support independent generalization claims."],
];
methods.getRange("A4:B12").values = methodRows;
methods.getRange("A4:A12").format = { fill: colors.lightBlue, font: { bold: true, color: colors.navy } };
methods.getRange("A4:B12").format.borders = { preset: "all", style: "thin", color: colors.border };
methods.getRange("A:B").format.columnWidth = 28;
methods.getRange("B:B").format.columnWidth = 80;
methods.getRange("A4:B12").format.wrapText = true;
methods.getRange("4:12").format.rowHeight = 34;

const inspect = await wb.inspect({
  kind: "table",
  range: "Dashboard!A1:P20",
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 16,
  maxChars: 5000,
});
console.log(inspect.ndjson);
const errors = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);

for (const sheetName of [
  "Dashboard", "Engineering_Cost", "Stock_Diagnostics", "Return_Groups",
  "Disagreement", "Sample_Details", "Methodology",
]) {
  const preview = await wb.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheetName}.png`), new Uint8Array(await preview.arrayBuffer()));
}

const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(outputPath);
console.log(JSON.stringify({ outputPath, sheets: wb.worksheets.items.map((sheet) => sheet.name) }));
