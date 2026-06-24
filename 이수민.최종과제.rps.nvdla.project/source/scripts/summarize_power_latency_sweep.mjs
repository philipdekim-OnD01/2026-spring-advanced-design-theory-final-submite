#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";

function parseCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (quoted && line[i + 1] === '"') {
        value += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      values.push(value);
      value = "";
    } else {
      value += char;
    }
  }
  values.push(value);
  return values;
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function csvEscape(value) {
  if (value === null || value === undefined) return "";
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function writeCsv(filePath, rows) {
  if (rows.length === 0) throw new Error(`No rows for ${filePath}`);
  const headers = Object.keys(rows[0]);
  const lines = [headers.join(",")];
  for (const row of rows) {
    lines.push(headers.map((header) => csvEscape(row[header])).join(","));
  }
  await fs.writeFile(filePath, `${lines.join("\n")}\n`);
}

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;

function stdev(values) {
  if (values.length < 2) return 0;
  const average = mean(values);
  return Math.sqrt(
    values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1),
  );
}

function round(value, digits = 9) {
  return Number(value.toFixed(digits));
}

function parsePowerLine(line) {
  const fields = {};
  for (const [name, pattern] of [
    ["vdd_gpu_soc_mw", /VDD_GPU_SOC\s+(\d+)mW\//],
    ["vdd_cpu_cv_mw", /VDD_CPU_CV\s+(\d+)mW\//],
    ["vin_sys_5v0_mw", /VIN_SYS_5V0\s+(\d+)mW\//],
    ["vddq_vdd2_1v8ao_mw", /VDDQ_VDD2_1V8AO\s+(\d+)mW\//],
    ["gr3d_freq_percent", /GR3D_FREQ\s+(\d+)%/],
  ]) {
    const match = line.match(pattern);
    if (!match) return null;
    fields[name] = Number(match[1]);
  }
  fields.compute_rails_mw = fields.vdd_gpu_soc_mw + fields.vdd_cpu_cv_mw;
  return fields;
}

async function parsePowerLog(filePath, trimStart, trimEnd) {
  const lines = (await fs.readFile(filePath, "utf8")).trim().split(/\r?\n/);
  const selected = lines.slice(trimStart, lines.length - trimEnd).map(parsePowerLine).filter(Boolean);
  if (selected.length === 0) throw new Error(`No power samples parsed: ${filePath}`);
  const result = { samples: selected.length };
  for (const field of Object.keys(selected[0])) {
    result[field] = mean(selected.map((sample) => sample[field]));
  }
  return result;
}

async function parseTrtexec(filePath) {
  const text = await fs.readFile(filePath, "utf8");
  if (!text.includes("&&&& PASSED")) throw new Error(`trtexec did not pass: ${filePath}`);
  const throughput = text.match(/Throughput:\s+([\d.]+)\s+qps/);
  const latency = text.match(/Latency: min = .*?mean = ([\d.]+) ms/);
  if (!throughput || !latency) throw new Error(`Missing trtexec metrics: ${filePath}`);
  return {
    throughput_qps: Number(throughput[1]),
    trtexec_latency_ms: Number(latency[1]),
  };
}

const [rawDir, nativeSummaryPath, outputDir] = process.argv.slice(2);
if (!rawDir || !nativeSummaryPath || !outputDir) {
  console.error("Usage: summarize_power_latency_sweep.mjs RAW_DIR NATIVE_SUMMARY_CSV OUTPUT_DIR");
  process.exit(2);
}

await fs.mkdir(outputDir, { recursive: true });
const nativeRows = parseCsv(await fs.readFile(nativeSummaryPath, "utf8"));
const nativeByWidth = new Map(nativeRows.map((row) => [Number(row.width_mult), row]));
const fileNames = await fs.readdir(rawDir);
const trtexecFiles = fileNames.filter((name) => name.endsWith("_trtexec.log")).sort();
const runRows = [];

for (const fileName of trtexecFiles) {
  const match = fileName.match(/^r(\d+)_s(\d+)_(w\d+_\d+)_(gpu|dla0)_trtexec\.log$/);
  if (!match) throw new Error(`Unexpected trtexec file name: ${fileName}`);
  const [, repeatText, sequenceText, model, device] = match;
  const widthMult = Number(model.slice(1).replace("_", "."));
  const native = nativeByWidth.get(widthMult);
  if (!native) throw new Error(`No native sweep metadata for ${model}`);
  const stem = fileName.replace("_trtexec.log", "");
  const idle = await parsePowerLog(path.join(rawDir, `${stem}_idle.log`), 5, 5);
  const load = await parsePowerLog(path.join(rawDir, `${stem}_load.log`), 10, 10);
  const trtexec = await parseTrtexec(path.join(rawDir, fileName));
  const e2eLatency = Number(
    device === "gpu" ? native.gpu_latency_ms_run_mean : native.strict_dla_latency_ms_run_mean,
  );
  const row = {
    repeat: Number(repeatText),
    sequence: Number(sequenceText),
    model,
    width_mult: widthMult,
    param_count: Number(native.param_count),
    macs_m: Number(native.macs_m),
    device,
    idle_samples: idle.samples,
    load_samples: load.samples,
    idle_vdd_gpu_soc_w: round(idle.vdd_gpu_soc_mw / 1000),
    load_vdd_gpu_soc_w: round(load.vdd_gpu_soc_mw / 1000),
    delta_vdd_gpu_soc_w: round((load.vdd_gpu_soc_mw - idle.vdd_gpu_soc_mw) / 1000),
    idle_vdd_cpu_cv_w: round(idle.vdd_cpu_cv_mw / 1000),
    load_vdd_cpu_cv_w: round(load.vdd_cpu_cv_mw / 1000),
    delta_vdd_cpu_cv_w: round((load.vdd_cpu_cv_mw - idle.vdd_cpu_cv_mw) / 1000),
    idle_compute_rails_w: round(idle.compute_rails_mw / 1000),
    load_compute_rails_w: round(load.compute_rails_mw / 1000),
    delta_compute_rails_w: round((load.compute_rails_mw - idle.compute_rails_mw) / 1000),
    load_vin_sys_5v0_w: round(load.vin_sys_5v0_mw / 1000),
    load_gr3d_freq_percent: round(load.gr3d_freq_percent),
    throughput_qps: trtexec.throughput_qps,
    trtexec_latency_ms: trtexec.trtexec_latency_ms,
    e2e_latency_ms: e2eLatency,
    load_energy_mj_per_inference: round(load.compute_rails_mw / trtexec.throughput_qps),
    dynamic_energy_mj_per_inference: round(
      (load.compute_rails_mw - idle.compute_rails_mw) / trtexec.throughput_qps,
    ),
    qps_per_load_watt: round(trtexec.throughput_qps / (load.compute_rails_mw / 1000)),
    qps_per_dynamic_watt: round(
      trtexec.throughput_qps / ((load.compute_rails_mw - idle.compute_rails_mw) / 1000),
    ),
    e2e_load_energy_proxy_mj: round((load.compute_rails_mw / 1000) * e2eLatency),
    e2e_dynamic_energy_proxy_mj: round(
      ((load.compute_rails_mw - idle.compute_rails_mw) / 1000) * e2eLatency,
    ),
  };
  runRows.push(row);
}

runRows.sort((a, b) => a.width_mult - b.width_mult || a.device.localeCompare(b.device) || a.repeat - b.repeat);
await writeCsv(path.join(outputDir, "power_latency_runs.csv"), runRows);

const metricNames = [
  "idle_vdd_gpu_soc_w", "load_vdd_gpu_soc_w", "delta_vdd_gpu_soc_w",
  "idle_vdd_cpu_cv_w", "load_vdd_cpu_cv_w", "delta_vdd_cpu_cv_w",
  "idle_compute_rails_w", "load_compute_rails_w", "delta_compute_rails_w",
  "load_vin_sys_5v0_w", "load_gr3d_freq_percent", "throughput_qps",
  "trtexec_latency_ms", "load_energy_mj_per_inference",
  "dynamic_energy_mj_per_inference", "qps_per_load_watt",
  "qps_per_dynamic_watt", "e2e_load_energy_proxy_mj",
  "e2e_dynamic_energy_proxy_mj",
];
const summaryRows = [];
for (const widthMult of [...nativeByWidth.keys()].sort((a, b) => a - b)) {
  for (const device of ["gpu", "dla0"]) {
    const selected = runRows.filter((row) => row.width_mult === widthMult && row.device === device);
    if (selected.length !== 3) throw new Error(`Expected 3 runs for w${widthMult} ${device}`);
    const summary = {
      model: selected[0].model,
      width_mult: widthMult,
      param_count: selected[0].param_count,
      macs_m: selected[0].macs_m,
      device,
      repeats: selected.length,
      e2e_latency_ms: selected[0].e2e_latency_ms,
    };
    for (const metric of metricNames) {
      const values = selected.map((row) => row[metric]);
      summary[`${metric}_mean`] = round(mean(values));
      summary[`${metric}_stdev`] = round(stdev(values));
    }
    summaryRows.push(summary);
  }
}
await writeCsv(path.join(outputDir, "power_latency_summary.csv"), summaryRows);

const comparisonRows = [];
for (const widthMult of [...nativeByWidth.keys()].sort((a, b) => a - b)) {
  const gpu = summaryRows.find((row) => row.width_mult === widthMult && row.device === "gpu");
  const dla = summaryRows.find((row) => row.width_mult === widthMult && row.device === "dla0");
  const comparison = {
    model: gpu.model,
    width_mult: widthMult,
    param_count: gpu.param_count,
    macs_m: gpu.macs_m,
    gpu_load_power_w: gpu.load_compute_rails_w_mean,
    dla_load_power_w: dla.load_compute_rails_w_mean,
    dla_load_power_saving_w: round(gpu.load_compute_rails_w_mean - dla.load_compute_rails_w_mean),
    dla_load_power_saving_percent: round(
      (gpu.load_compute_rails_w_mean - dla.load_compute_rails_w_mean) / gpu.load_compute_rails_w_mean,
    ),
    gpu_dynamic_power_w: gpu.delta_compute_rails_w_mean,
    dla_dynamic_power_w: dla.delta_compute_rails_w_mean,
    dla_dynamic_power_saving_percent: round(
      (gpu.delta_compute_rails_w_mean - dla.delta_compute_rails_w_mean) / gpu.delta_compute_rails_w_mean,
    ),
    gpu_trtexec_latency_ms: gpu.trtexec_latency_ms_mean,
    dla_trtexec_latency_ms: dla.trtexec_latency_ms_mean,
    dla_over_gpu_trtexec_latency_ratio: round(
      dla.trtexec_latency_ms_mean / gpu.trtexec_latency_ms_mean,
    ),
    gpu_e2e_latency_ms: gpu.e2e_latency_ms,
    dla_e2e_latency_ms: dla.e2e_latency_ms,
    dla_over_gpu_e2e_latency_ratio: round(dla.e2e_latency_ms / gpu.e2e_latency_ms),
    gpu_load_energy_mj_per_inference: gpu.load_energy_mj_per_inference_mean,
    dla_load_energy_mj_per_inference: dla.load_energy_mj_per_inference_mean,
    dla_load_energy_change_percent: round(
      (dla.load_energy_mj_per_inference_mean - gpu.load_energy_mj_per_inference_mean)
        / gpu.load_energy_mj_per_inference_mean,
    ),
    gpu_dynamic_energy_mj_per_inference: gpu.dynamic_energy_mj_per_inference_mean,
    dla_dynamic_energy_mj_per_inference: dla.dynamic_energy_mj_per_inference_mean,
    dla_dynamic_energy_change_percent: round(
      (dla.dynamic_energy_mj_per_inference_mean - gpu.dynamic_energy_mj_per_inference_mean)
        / gpu.dynamic_energy_mj_per_inference_mean,
    ),
    gpu_qps_per_load_watt: gpu.qps_per_load_watt_mean,
    dla_qps_per_load_watt: dla.qps_per_load_watt_mean,
    dla_qps_per_watt_change_percent: round(
      (dla.qps_per_load_watt_mean - gpu.qps_per_load_watt_mean) / gpu.qps_per_load_watt_mean,
    ),
  };
  comparisonRows.push(comparison);
}
await writeCsv(path.join(outputDir, "power_latency_comparison.csv"), comparisonRows);

console.log(JSON.stringify({ runs: runRows.length, summaries: summaryRows.length, comparisons: comparisonRows }, null, 2));
