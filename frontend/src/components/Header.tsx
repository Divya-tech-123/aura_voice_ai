import React from 'react';
import { Menu, Settings, Sparkles } from 'lucide-react';

interface HeaderProps {
  onToggleSidebar: () => void;
  onOpenSettings?: () => void;
  status?: 'ready' | 'thinking' | 'error';
}

export const Header: React.FC<HeaderProps> = ({
  onToggleSidebar,
  onOpenSettings,
  status = 'ready',
}) => {
  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/80 backdrop-blur-md px-4 sm:px-6 flex items-center justify-between shrink-0 select-none z-10">
      <div className="flex items-center gap-3">
        {/* Mobile menu button */}
        <button
          type="button"
          onClick={onToggleSidebar}
          className="p-2 -ml-2 rounded-xl text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors lg:hidden focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
          aria-label="Toggle navigation menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* Brand */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center justify-center w-8 h-8 rounded-xl bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 shadow-sm">
            <Sparkles className="w-4 h-4" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-bold text-base text-slate-100 tracking-tight">
              AURA
            </span>
            <span className="hidden sm:inline-block text-[11px] font-medium text-slate-400">
              AI Unified Response Assistant
            </span>
          </div>
        </div>
      </div>

      {/* Right controls */}
      <div className="flex items-center gap-3">
        {/* Status indicator */}
        <div
          role="status"
          className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/90 border border-slate-700/60 text-xs shadow-sm"
        >
          {status === 'thinking' ? (
            <>
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-400"></span>
              </span>
              <span className="font-medium text-[11px] text-cyan-300">Thinking...</span>
            </>
          ) : status === 'error' ? (
            <>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-500"></span>
              <span className="font-medium text-[11px] text-rose-300">Issue</span>
            </>
          ) : (
            <>
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="font-medium text-[11px] text-slate-300">Ready</span>
            </>
          )}
        </div>

        {/* Settings button */}
        {onOpenSettings && (
          <button
            type="button"
            onClick={onOpenSettings}
            className="p-2 rounded-xl text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500/40 cursor-pointer"
            aria-label="Open settings"
            title="Settings"
          >
            <Settings className="w-4 h-4" />
          </button>
        )}
      </div>
    </header>
  );
};

export default Header;
