/**
 * 参考数据库（偏差数据库）选择器。
 *
 * 聚类展示与聚类测试共用同一个控件，保证两页对「本次用哪个库」的口径一致：
 * 选项文案、默认库解释、不可用原因都由这里产出，页面不再各写一套。
 *
 * 两个关键取舍：
 *
 * 1. **只要引擎就绪就渲染**，而不是「选了检索增强 profile 才出现」。整块消失会让
 *    用户以为系统根本没有这个能力；现在控件始终可见、状态明确。
 * 2. **未指定 profile 时也可选**。后端在收到 `knowledgeBaseId` 时会自动把候选
 *    收窄到检索增强 profile（见 `_resolve_profile`），所以"先选库、profile 交给
 *    自动挑选"是一条真实可用的路径，不该在界面上被禁掉。
 */

import type { ClusteringKnowledgeBaseInfo, ClusteringProfileInfo } from '../../shared/clustering';

/** 把知识库标识翻译成可读名称；找不到就退回标识本身，绝不静默吞掉。 */
export function knowledgeBaseName(
  knowledgeBases: ClusteringKnowledgeBaseInfo[],
  knowledgeBaseId: string | null | undefined,
): string {
  if (!knowledgeBaseId) return '';
  return (
    knowledgeBases.find((kb) => kb.knowledgeBaseId === knowledgeBaseId)?.displayName ?? knowledgeBaseId
  );
}

/** profile 是否做检索增强（只有这类 profile 才吃参考数据库）。 */
export function supportsRetrieval(profile: ClusteringProfileInfo | null | undefined): boolean {
  return Number(profile?.features?.n_results ?? 0) > 0;
}

export interface KnowledgeBasePickerProps {
  /** 引擎里现有知识库清单。 */
  knowledgeBases: ClusteringKnowledgeBaseInfo[];
  /** 本次显式选择的知识库；空串表示按 profile 默认。 */
  value: string;
  onChange: (next: string) => void;
  /** 当前所选 profile；缺省表示尚未显式选择（自动挑选）。 */
  profile?: ClusteringProfileInfo | null;
  /** 运行中或提交中禁用交互。 */
  disabled?: boolean;
  /** 清单拉取失败等外部原因；有值时优先作为说明文案。 */
  unavailableReason?: string;
}

export function KnowledgeBasePicker({
  knowledgeBases,
  value,
  onChange,
  profile,
  disabled = false,
  unavailableReason,
}: KnowledgeBasePickerProps) {
  const retrieval = supportsRetrieval(profile);
  /** 未指定 profile 时交给后端自动挑检索增强 profile；显式选了纯向量 profile 才禁用。 */
  const selectable = !profile || retrieval;

  const defaultId = profile?.knowledgeBaseId ?? '';
  const defaultName = knowledgeBaseName(knowledgeBases, defaultId);

  // 生效的库 = 显式选择 > profile 默认；两者都没有时才算「未指定」。
  const effectiveId = value || defaultId;
  const effective = knowledgeBases.find((kb) => kb.knowledgeBaseId === effectiveId) ?? null;

  let hint: string;
  if (profile && !retrieval) {
    hint = '当前 profile 不做检索增强，参考数据库不参与本次计算；换成带「检索增强」标记的 profile 后即可指定。';
  } else if (unavailableReason) {
    hint = unavailableReason;
  } else if (knowledgeBases.length === 0) {
    hint = '引擎里还没有可选的偏差数据库，请先到「偏差数据库」页生成一个。';
  } else if (!profile) {
    hint = effective
      ? `未指定 profile：后端会自动挑选检索增强 profile，并使用「${effective.displayName}」。`
      : '未指定 profile 时由后端自动挑选；也可以直接在这里选一个库，后端会自动挑检索增强 profile 来用它。';
  } else if (effective) {
    hint = `本次检索增强将使用「${effective.displayName}」（${effective.entryCount.toLocaleString('zh-CN')} 条${
      effective.status === 'completed' ? '' : ` · 状态 ${effective.status}`
    }）。`;
  } else if (defaultId) {
    hint = `profile 默认的库「${defaultId}」不在当前清单里，本次可能不可执行；请在下方另选一个库。`;
  } else {
    hint = '当前 profile 没有默认库，请明确选择一个参考数据库。';
  }

  const badge = profile && !retrieval ? '当前不生效' : retrieval ? '本次生效' : '自动匹配';

  return (
    <div className={`kb-picker ${selectable ? 'is-active' : ''}`}>
      <label className="kb-picker-field">
        <span className="kb-picker-title">
          参考数据库
          <span className={`kb-picker-badge ${selectable ? 'is-on' : ''}`}>{badge}</span>
        </span>
        <select
          className="select"
          aria-label="参考数据库"
          value={value}
          disabled={disabled || !selectable}
          onChange={(event) => onChange(event.target.value)}
          title={selectable ? '选择本次检索增强所用的偏差数据库' : '当前 profile 不做检索增强'}
        >
          <option value="">{defaultId ? `按 profile 默认（${defaultName}）` : '按 profile 默认'}</option>
          {knowledgeBases.map((kb) => (
            <option key={kb.knowledgeBaseId} value={kb.knowledgeBaseId}>
              {kb.displayName}（{kb.entryCount.toLocaleString('zh-CN')} 条
              {defaultId === kb.knowledgeBaseId ? ' · profile 默认' : ''}）
            </option>
          ))}
        </select>
      </label>
      <p className={`kb-picker-hint ${selectable ? '' : 'is-muted'}`}>{hint}</p>
    </div>
  );
}
