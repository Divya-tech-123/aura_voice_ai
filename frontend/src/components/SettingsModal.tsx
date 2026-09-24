import React, { useState, useEffect } from 'react';
import { X, Moon, Sun, Monitor, Mic, Volume2, Trash2, AlertTriangle, ShieldCheck } from 'lucide-react';
import { AppSettings, ThemePreference } from '../types';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: AppSettings;
  onUpdateSettings: (newSettings: Partial<AppSettings>) => void;
  onClearConversation: () => void;
  hasMessages: boolean;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onUpdateSettings,
  onClearConversation,
  hasMessages,
}) => {
  const [showClearConfirm, setShowClearConfirm] = useState(false);

  // Close on Escape key press
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (showClearConfirm) {
          setShowClearConfirm(false);
        } else {
          onClose();
        }
      }
    };

    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, showClearConfirm, onClose]);

  // Reset confirmation state when modal is opened/closed
  useEffect(() => {
    if (!isOpen) {
      setShowClearConfirm(false);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleClearConfirmed = () => {
    onClearConversation();
    setShowClearConfirm(false);
    onClose();
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="settings-dialog-title"
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in"
    >
      {/* Backdrop click dismiss */}
      <div
        className="fixed inset-0"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Modal Dialog Container */}
      <div className="relative w-full max-w-lg rounded-2xl bg-slate-900 border border-slate-800 text-slate-100 shadow-2xl z-10 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800">
          <div>
            <h2 id="settings-dialog-title" className="text-lg font-semibold tracking-tight text-white">
              Settings
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Customize your AURA workspace experience
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close settings dialog"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-6 max-h-[80vh] overflow-y-auto">
          {/* Section 1: Appearance / Theme Preference */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2.5">
              Theme Preference
            </label>
            <div className="grid grid-cols-3 gap-2.5">
              {[
                { id: 'dark', label: 'Dark', icon: Moon },
                { id: 'light', label: 'Light', icon: Sun },
                { id: 'system', label: 'System', icon: Monitor },
              ].map((themeOption) => {
                const Icon = themeOption.icon;
                const isSelected = settings.theme === themeOption.id;
                return (
                  <button
                    key={themeOption.id}
                    type="button"
                    onClick={() => onUpdateSettings({ theme: themeOption.id as ThemePreference })}
                    aria-pressed={isSelected}
                    className={`flex flex-col items-center justify-center p-3 rounded-xl border text-sm font-medium transition-all focus:outline-none focus:ring-2 focus:ring-cyan-500/50 ${
                      isSelected
                        ? 'bg-cyan-950/60 border-cyan-500/80 text-cyan-300 shadow-sm'
                        : 'bg-slate-800/60 border-slate-700/60 text-slate-300 hover:bg-slate-800 hover:border-slate-600'
                    }`}
                  >
                    <Icon className={`w-5 h-5 mb-1.5 ${isSelected ? 'text-cyan-400' : 'text-slate-400'}`} />
                    <span>{themeOption.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Section 2: Voice & Audio Settings */}
          <div className="border-t border-slate-800 pt-5">
            <span className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
              Voice &amp; Audio
            </span>
            <div className="space-y-3.5">
              {/* Voice Input Toggle */}
              <div className="flex items-center justify-between p-3.5 rounded-xl bg-slate-800/40 border border-slate-700/50">
                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-slate-800 border border-slate-700 text-cyan-400 mt-0.5">
                    <Mic className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-slate-200">Enable Voice Input</div>
                    <div className="text-xs text-slate-400 mt-0.5">
                      Allow speaking into microphone for speech recognition
                    </div>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer ml-3">
                  <input
                    type="checkbox"
                    checked={settings.voiceEnabled}
                    onChange={(e) => onUpdateSettings({ voiceEnabled: e.target.checked })}
                    className="sr-only peer"
                    aria-label="Toggle voice input"
                  />
                  <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan-600"></div>
                </label>
              </div>

              {/* Auto-play Responses Toggle */}
              <div className="flex items-center justify-between p-3.5 rounded-xl bg-slate-800/40 border border-slate-700/50">
                <div className="flex items-start gap-3">
                  <div className="p-2 rounded-lg bg-slate-800 border border-slate-700 text-cyan-400 mt-0.5">
                    <Volume2 className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-slate-200">Auto-play Voice Responses</div>
                    <div className="text-xs text-slate-400 mt-0.5">
                      Automatically play speech audio when AURA speaks
                    </div>
                  </div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer ml-3">
                  <input
                    type="checkbox"
                    checked={settings.autoPlayVoice}
                    disabled={!settings.voiceEnabled}
                    onChange={(e) => onUpdateSettings({ autoPlayVoice: e.target.checked })}
                    className="sr-only peer disabled:cursor-not-allowed"
                    aria-label="Toggle auto-play voice responses"
                  />
                  <div className="w-11 h-6 bg-slate-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cyan-600 peer-disabled:opacity-40"></div>
                </label>
              </div>
            </div>
          </div>

          {/* Section 3: Conversation Management & Safe Clear */}
          <div className="border-t border-slate-800 pt-5">
            <span className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
              Conversation Management
            </span>

            {!showClearConfirm ? (
              <div className="flex items-center justify-between p-3.5 rounded-xl bg-slate-800/40 border border-slate-700/50">
                <div>
                  <div className="text-sm font-medium text-slate-200">Clear Current Conversation</div>
                  <div className="text-xs text-slate-400 mt-0.5">
                    {hasMessages
                      ? 'Erase all messages in the active chat session'
                      : 'No messages in active session'}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setShowClearConfirm(true)}
                  disabled={!hasMessages}
                  className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-rose-950/60 hover:bg-rose-900/80 border border-rose-800/60 text-rose-300 text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-rose-500/50 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Clear</span>
                </button>
              </div>
            ) : (
              <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-700/60 text-rose-200 space-y-3 animate-fade-in">
                <div className="flex items-start gap-2.5">
                  <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
                  <div>
                    <div className="text-sm font-semibold text-rose-200">
                      Clear Conversation History?
                    </div>
                    <div className="text-xs text-rose-300/80 mt-1 leading-relaxed">
                      This will permanently clear all messages in this conversation. This action cannot be undone.
                    </div>
                  </div>
                </div>
                <div className="flex items-center justify-end gap-2 pt-1">
                  <button
                    type="button"
                    onClick={() => setShowClearConfirm(false)}
                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium transition-colors focus:outline-none focus:ring-1 focus:ring-slate-600"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={handleClearConfirmed}
                    className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-rose-500"
                  >
                    Yes, Clear Conversation
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Section 4: Privacy & System Info */}
          <div className="border-t border-slate-800 pt-5">
            <div className="flex items-start gap-3 p-3.5 rounded-xl bg-slate-950/50 border border-slate-800 text-xs text-slate-400">
              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <div className="font-medium text-slate-300">Privacy &amp; Local Security</div>
                <div className="text-[11px] text-slate-400 leading-relaxed">
                  AURA operates with local API endpoints. No API keys or internal stack traces are displayed or leaked to third parties.
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3.5 border-t border-slate-800 bg-slate-950/50 flex items-center justify-between text-xs text-slate-400">
          <span>AURA v0.1.0 • Professional UI</span>
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-100 text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};

export default SettingsModal;
