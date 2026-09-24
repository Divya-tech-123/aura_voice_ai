import React from 'react';
import { Plus, MessageSquare, Settings, Sparkles, X, Trash2 } from 'lucide-react';
import { DocumentUpload } from './DocumentUpload';
import { ConversationSession } from '../types';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onNewChat: () => void;
  conversations?: ConversationSession[];
  activeConversationId?: string | null;
  onSelectConversation?: (id: string) => void;
  onDeleteConversation?: (id: string) => void;
  onOpenSettings: () => void;
  documentUploadTriggerRef?: React.MutableRefObject<(() => void) | null>;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  onNewChat,
  conversations = [],
  activeConversationId,
  onSelectConversation,
  onDeleteConversation,
  onOpenSettings,
  documentUploadTriggerRef,
}) => {
  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm lg:hidden transition-opacity"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar Aside */}
      <aside
        className={`fixed lg:static top-0 left-0 z-50 h-full w-72 flex flex-col bg-slate-900 border-r border-slate-800 text-slate-200 transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
        aria-label="Navigation sidebar"
      >
        {/* Logo & Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 shadow-sm">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-base text-slate-100 tracking-tight">
                  AURA
                </span>
                <span className="text-[10px] uppercase font-semibold tracking-wider px-1.5 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800/60">
                  v0.1.0
                </span>
              </div>
              <p className="text-xs text-slate-400">AI Assistant</p>
            </div>
          </div>

          {/* Close button for mobile drawer */}
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors lg:hidden focus:outline-none focus:ring-2 focus:ring-cyan-500/40"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-4 pb-2">
          <button
            type="button"
            onClick={() => {
              onNewChat();
              if (window.innerWidth < 1024) onClose();
            }}
            className="w-full flex items-center justify-center gap-2.5 py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700/90 border border-slate-700/70 text-slate-100 text-sm font-medium transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/40 cursor-pointer"
          >
            <Plus className="w-4 h-4 text-cyan-400" />
            <span>New Chat</span>
          </button>
        </div>

        {/* Scrollable Body: Conversations & Document Upload */}
        <div className="flex-1 overflow-y-auto px-4 py-2 space-y-6">
          {/* Recent Conversations Section */}
          <div>
            <div className="px-1 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              Conversations
            </div>

            {/* Conversation list or Empty State */}
            {conversations.length === 0 ? (
              <div className="mt-1 px-3 py-4 rounded-xl border border-dashed border-slate-800 text-center bg-slate-900/40">
                <MessageSquare className="w-5 h-5 mx-auto text-slate-600 mb-1.5 stroke-[1.5]" />
                <p className="text-xs text-slate-300 font-medium">No conversations yet</p>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Start a new session to begin
                </p>
              </div>
            ) : (
              <div className="flex flex-col gap-1 max-h-56 overflow-y-auto pr-0.5">
                {conversations.map((conv) => {
                  const isActive = conv.id === activeConversationId;
                  return (
                    <div
                      key={conv.id}
                      className={`group flex items-center justify-between gap-2 px-3 py-2 rounded-xl text-xs font-medium transition-colors cursor-pointer ${
                        isActive
                          ? 'bg-slate-800 text-cyan-300 border border-cyan-500/30'
                          : 'text-slate-300 hover:bg-slate-800/60 hover:text-slate-100'
                      }`}
                      onClick={() => {
                        onSelectConversation?.(conv.id);
                        if (window.innerWidth < 1024) onClose();
                      }}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          onSelectConversation?.(conv.id);
                          if (window.innerWidth < 1024) onClose();
                        }
                      }}
                    >
                      <div className="flex items-center gap-2 min-w-0 flex-1">
                        <MessageSquare className={`w-3.5 h-3.5 shrink-0 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
                        <span className="truncate">{conv.title || 'Untitled Conversation'}</span>
                      </div>

                      {/* Delete conversation button */}
                      {onDeleteConversation && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteConversation(conv.id);
                          }}
                          className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-slate-700/80 text-slate-400 hover:text-rose-400 transition-all"
                          title="Delete conversation"
                          aria-label={`Delete conversation ${conv.title}`}
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* Knowledge Documents Section (RAG) */}
          <div className="border-t border-slate-800 pt-4">
            <DocumentUpload triggerRef={documentUploadTriggerRef} />
          </div>
        </div>

        {/* Settings Footer */}
        <div className="p-4 border-t border-slate-800">
          <button
            type="button"
            onClick={() => {
              onOpenSettings();
              if (window.innerWidth < 1024) onClose();
            }}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-slate-300 hover:text-slate-100 hover:bg-slate-800/80 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500/40 cursor-pointer"
          >
            <Settings className="w-4 h-4 text-slate-400" />
            <span>Settings</span>
          </button>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
