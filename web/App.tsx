import { useCallback, useEffect, useState } from 'react';
import { AppSidebar } from './components/AppSidebar';
import { TopBar } from './components/TopBar';
import { fetchModelInfo } from './api/model';
import { applyTheme, getTheme, setTheme, type ThemeId } from './lib/theme';
import { useHashRoute } from './lib/useHashRoute';
import { CameraDiscover } from './pages/CameraDiscover';
import { ClusteringOverview } from './pages/ClusteringOverview';
import { ClusteringTest } from './pages/ClusteringTest';
import { Graph } from './pages/Graph';
import { Knowledge } from './pages/Knowledge';
import { KnowledgeBase } from './pages/KnowledgeBase';
import { ModelConfig } from './pages/ModelConfig';
import { Prevent } from './pages/Prevent';
import { Records } from './pages/Records';
import { HazardTest } from './pages/HazardTest';
import { RobotDiscover } from './pages/RobotDiscover';
import { Rules } from './pages/Rules';
import { Settings } from './pages/Settings';
import { Workbench } from './pages/Workbench';
import { TokenUsagePage } from './pages/TokenUsage';
import './styles.css';

export function App() {
  const [route, navigate] = useHashRoute();
  const [modelName, setModelName] = useState('');
  const [theme, setThemeState] = useState<ThemeId>(() => {
    const t = getTheme();
    return t;
  });

  // 主题初始化
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const applyModelInfo = useCallback((i: { model: string }) => {
    if (i.model) setModelName(i.model);
  }, []);

  const refreshModelInfo = useCallback(() => {
    fetchModelInfo()
      .then(applyModelInfo)
      .catch(() => undefined);
  }, [applyModelInfo]);

  // 初始加载 + 路由切换时刷新模型信息（模型配置保存后回到工作台能同步顶栏模式）
  useEffect(() => {
    refreshModelInfo();
  }, [route, refreshModelInfo]);

  const pickTheme = useCallback(
    (id: ThemeId) => {
      setTheme(id);
      setThemeState(id);
    },
    []
  );

  let page;
  const r = route;
  if (r.startsWith('/discover/robot')) page = <RobotDiscover />;
  else if (r.startsWith('/discover')) page = <CameraDiscover />;
  else if (r === '/prevent') page = <Prevent />;
  else if (r === '/rules') page = <Rules />;
  else if (r.startsWith('/graph')) page = <Graph />;
  else if (r === '/knowledge') page = <Knowledge initialTab="standard" />;
  else if (r === '/cases') page = <Knowledge initialTab="case" />;
  else if (r === '/records') page = <Records onOpenRecord={() => navigate('/')} />;
  else if (r === '/hazard-test') page = <HazardTest />;
  // 偏差数据库必须先于通用 /clustering 分支匹配
  else if (r.startsWith('/clustering/knowledge-base')) page = <KnowledgeBase />;
  else if (r.startsWith('/clustering/test')) page = <ClusteringTest />;
  else if (r.startsWith('/clustering')) page = <ClusteringOverview />;
  else if (r === '/tokens') page = <TokenUsagePage />;
  else if (r === '/model-config') page = <ModelConfig onConfigChanged={refreshModelInfo} />;
  else if (r === '/settings') page = <Settings theme={theme} onPickTheme={pickTheme} />;
  else page = <Workbench modelName={modelName} />;

  return (
    <div className="app">
      <TopBar
        modelName={modelName}
        theme={theme}
        onPickTheme={pickTheme}
      />
      <div className="shell">
        <AppSidebar route={route} onNavigate={navigate} />
        <main className="main">{page}</main>
      </div>
    </div>
  );
}
