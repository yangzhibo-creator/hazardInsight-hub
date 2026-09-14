/**
 * 聚类告警码分流。
 *
 * 有些告警码是给开发/现场排查看的**原始英文码**，直接写在结果卡片上会被观众
 * 当成报错，或者把"预期行为"误读成异常。因此界面上只保留能读懂的中文提示，
 * 原始码改走浏览器控制台（开发者工具的 Console 即"调试窗口"）。
 *
 * 数据本身不做任何删改：控制台里始终能看到完整原始码，随时可查。
 */

/**
 * 只在调试窗口输出、不在界面展示的告警码。
 *
 * * `POLARITY_GUARD_TRIGGERED` —— SPEAR 极性保真护栏命中，属于**预期行为**
 *   （护栏拦下并救回了语义反转的条目），代码本身对观众没有意义；
 * * `SPEAR_VIZ_NOT_COMPUTED` —— SPEAR 路径不生成二维坐标，结果区已有专门的
 *   中文说明，再重复一行英文码只会让人以为出错了。
 */
export const DEBUG_ONLY_CLUSTERING_WARNINGS: readonly string[] = [
  'POLARITY_GUARD_TRIGGERED',
  'SPEAR_VIZ_NOT_COMPUTED',
];

const DEBUG_ONLY = new Set<string>(DEBUG_ONLY_CLUSTERING_WARNINGS);

export interface ClusteringWarningSplit {
  /** 可以在界面上展示的告警（保持原有顺序）。 */
  visible: string[];
  /** 只写控制台的告警码。 */
  debug: string[];
}

/**
 * 按告警码分流，并去重、去掉空白项。
 *
 * 去重是必要的：同一个码可能同时出现在作业级 warnings 与结果 summary.warnings
 * 里，直接合并渲染会让 React 因为 key 重复而告警。
 */
export function splitClusteringWarnings(warnings?: readonly string[] | null): ClusteringWarningSplit {
  const visible: string[] = [];
  const debug: string[] = [];
  const seen = new Set<string>();
  for (const warning of warnings ?? []) {
    if (typeof warning !== 'string' || !warning.trim() || seen.has(warning)) continue;
    seen.add(warning);
    if (DEBUG_ONLY.has(warning)) debug.push(warning);
    else visible.push(warning);
  }
  return { visible, debug };
}

/** 把被隐藏的告警码写到浏览器控制台；没有可隐藏项时保持安静，不刷屏。 */
export function logDebugWarnings(scope: string, warnings: readonly string[]): void {
  if (!warnings.length) return;
  const prefix = scope ? `[clustering] ${scope} ` : '[clustering] ';
  console.debug(`${prefix}调试告警码：${warnings.join(', ')}`, warnings);
}
