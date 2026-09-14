import { useEffect, useState, type ReactElement } from 'react';
import { IconBook, IconCamera, IconChevronDown, IconScatter, IconSettings, IconShieldCheck } from './icons';

export interface SubItem {
  id: string;
  label: string;
  route: string;
}

/** 原「一级菜单」，如今降为目录下的二级分组 */
export interface MenuGroup {
  id: string;
  label: string;
  items: SubItem[];
}

/** 新增的一级目录：隐患 / 聚类 / 设置 */
export interface MenuSection {
  id: string;
  label: string;
  icon: ReactElement;
  /** 无二级分组的目录：页面直接挂在目录下（如「聚类」） */
  items?: SubItem[];
  /** 二级分组（原有的「一级菜单」，如 隐患发现 / 隐患识别 / …） */
  groups?: MenuGroup[];
}

/**
 * 导航为「目录 → 分组 → 页面」三级，保留原有两级结构并在其上新增一层目录：
 *   隐患管理（目录，含二级分组）
 *     └ 隐患识别（分组）
 *         └ 图片智能识别（页面）
 *   隐患发现 / 隐患知识库 / 偏差聚类（目录，无二级分组，页面直接挂载）
 *     └ 固定摄像头抓隐患、安全知识库、聚类展示、偏差数据库 …（页面）
 * 分组只含单个页面时渲染为直接跳转项，不再多套一层。
 */
const NAV: MenuSection[] = [
  {
    id: 'discover',
    label: '隐患发现',
    icon: <IconCamera />,
    items: [
      { id: 'discover-camera', label: '固定摄像头抓隐患', route: '/discover/camera' },
      { id: 'discover-robot', label: '具身智能机器人寻隐患', route: '/discover/robot' },
    ],
  },
  {
    id: 'hazard',
    label: '隐患管理',
    icon: <IconShieldCheck />,
    groups: [
      { id: 'analyze', label: '隐患识别', items: [{ id: 'analyze-image', label: '图片智能识别', route: '/' }] },
      { id: 'manage', label: '历史记录', items: [{ id: 'manage-records', label: '隐患记录', route: '/records' }] },
      { id: 'hazard-test', label: '隐患测试', items: [{ id: 'hazard-test-main', label: '隐患测试', route: '/hazard-test' }] },
      { id: 'prevent', label: '隐患防控', items: [{ id: 'prevent-ledger', label: '隐患台账', route: '/prevent' }] },
    ],
  },
  {
    id: 'kb',
    label: '隐患知识库',
    icon: <IconBook />,
    items: [
      { id: 'kb-standards', label: '安全知识库', route: '/knowledge' },
      { id: 'kb-cases', label: '历史案例', route: '/cases' },
      { id: 'kb-rules', label: '定级规则', route: '/rules' },
      { id: 'kb-graph', label: '知识图谱', route: '/graph' },
    ],
  },
  {
    id: 'clustering',
    label: '偏差聚类',
    icon: <IconScatter />,
    items: [
      { id: 'clustering-overview', label: '聚类展示', route: '/clustering/overview' },
      { id: 'clustering-test', label: '聚类测试', route: '/clustering/test' },
      { id: 'clustering-knowledge-base', label: '偏差数据库', route: '/clustering/knowledge-base' },
    ],
  },
  {
    id: 'settings',
    label: '设置',
    icon: <IconSettings />,
    groups: [
      { id: 'model-config', label: '模型配置', items: [{ id: 'model-config-main', label: '模型配置', route: '/model-config' }] },
      { id: 'tokens', label: 'Token 统计', items: [{ id: 'tokens-usage', label: 'Token 统计', route: '/tokens' }] },
      { id: 'settings', label: '系统设置', items: [{ id: 'settings-main', label: '系统设置', route: '/settings' }] },
    ],
  },
];

/** 根据路由定位所属二级分组（沿用原有规则，未变） */
export function activeGroupOf(route: string): string {
  if (route === '/hazard-test') return 'hazard-test';
  if (route.startsWith('/clustering')) return 'clustering';
  if (route.startsWith('/discover')) return 'discover';
  if (route === '/') return 'analyze';
  if (route.startsWith('/records')) return 'manage';
  if (route.startsWith('/prevent')) return 'prevent';
  if (route.startsWith('/knowledge') || route.startsWith('/cases') || route.startsWith('/rules') || route.startsWith('/graph')) return 'kb';
  if (route.startsWith('/tokens')) return 'tokens';
  if (route.startsWith('/model-config')) return 'model-config';
  if (route.startsWith('/settings')) return 'settings';
  return 'analyze';
}

/** 根据路由定位所属一级目录（由页面/分组归属反查，保持单一数据源） */
export function activeSectionOf(route: string): string {
  const group = activeGroupOf(route);
  for (const s of NAV) {
    if (s.items?.some((i) => i.route === route)) return s.id;
    if (s.groups?.some((g) => g.id === group)) return s.id;
  }
  return NAV[0].id;
}

export function AppSidebar({ route, onNavigate }: { route: string; onNavigate: (r: string) => void }) {
  const [openSections, setOpenSections] = useState<Set<string>>(() => new Set([activeSectionOf(locationHash())]));
  const [openGroups, setOpenGroups] = useState<Set<string>>(() => new Set([activeGroupOf(locationHash())]));

  // 路由变化时自动展开当前项所在的目录与分组
  useEffect(() => {
    setOpenSections((prev) => new Set([...prev, activeSectionOf(route)]));
    setOpenGroups((prev) => new Set([...prev, activeGroupOf(route)]));
  }, [route]);

  function toggle(set: Set<string>, id: string): Set<string> {
    const next = new Set(set);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    return next;
  }

  const currentSection = activeSectionOf(route);
  const currentGroup = activeGroupOf(route);

  return (
    <aside className="sidebar">
      <div className="side-title">业务功能</div>
      <nav className="side-nav">
        {NAV.map((sec) => {
          const secOpen = openSections.has(sec.id);
          return (
            <div key={sec.id} className="nav-section">
              <button
                className={`nav-item nav-l1 ${currentSection === sec.id ? 'active' : ''}`}
                onClick={() => setOpenSections((prev) => toggle(prev, sec.id))}
                aria-expanded={secOpen}
              >
                <span className="nav-ico">{sec.icon}</span>
                {sec.label}
                <span className={`nav-chev ${secOpen ? 'open' : ''}`}>
                  <IconChevronDown />
                </span>
              </button>

              {secOpen && (
                <div className="nav-sub nav-l2">
                  {/* 无二级分组的目录：页面直接挂在目录下 */}
                  {(sec.items ?? []).map((item) => (
                    <button
                      key={item.id}
                      className={`nav-subitem ${route === item.route ? 'active' : ''}`}
                      onClick={() => onNavigate(item.route)}
                    >
                      {item.label}
                    </button>
                  ))}
                  {(sec.groups ?? []).map((g) => {
                    // 分组只有一个页面时按原样直接跳转，不再多套一层
                    if (g.items.length === 1) {
                      const only = g.items[0];
                      return (
                        <button
                          key={g.id}
                          className={`nav-subitem ${route === only.route ? 'active' : ''}`}
                          onClick={() => onNavigate(only.route)}
                        >
                          {g.label}
                        </button>
                      );
                    }
                    const gOpen = openGroups.has(g.id);
                    return (
                      <div key={g.id} className="nav-group">
                        <button
                          className={`nav-subitem nav-subitem-head ${currentGroup === g.id ? 'active' : ''}`}
                          onClick={() => setOpenGroups((prev) => toggle(prev, g.id))}
                          aria-expanded={gOpen}
                        >
                          {g.label}
                          <span className={`nav-chev ${gOpen ? 'open' : ''}`}>
                            <IconChevronDown />
                          </span>
                        </button>
                        {gOpen && (
                          <div className="nav-sub nav-l3">
                            {g.items.map((item) => (
                              <button
                                key={item.id}
                                className={`nav-subitem ${route === item.route ? 'active' : ''}`}
                                onClick={() => onNavigate(item.route)}
                              >
                                {item.label}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </nav>
    </aside>
  );
}

function locationHash(): string {
  return typeof window === 'undefined' ? '/' : (window.location.hash.replace(/^#/, '') || '/');
}
