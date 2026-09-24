import React, { useEffect, useRef } from 'react';
import { WelcomeScreen } from './WelcomeScreen';
import { MessageInput, VoiceState } from './MessageInput';
import { AgentActivity } from './AgentActivity';
import { MarkdownRenderer } from './MarkdownRenderer';
import { Sparkles, User, AlertCircle, RefreshCw } from 'lucide-react';
import { ChatMessage } from '../types';

interface ChatAreaProps {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  onClearError?: () => void;
  onRetry?: () => void;
  inputValue: string;
  onInputChange: (val: string) => void;
  onSubmitMessage: (msg: string) => void;
  onSelectPrompt: (prompt: string) => void;
  voiceEnabled?: boolean;
  voiceState?: VoiceState;
  onVoiceClick?: () => void;
  onCancelVoice?: () => void;
  onAttachDocument?: () => void;
}

/**
 * Format timestamp nicely into a readable time string
 */
function formatMessageTime(timestamp: string | Date | undefined): string {
  if (!timestamp) return '';
  try {
    const d = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
    if (isNaN(d.getTime())) return '';
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return '';
  }
}

export const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  isLoading,
  error,
  onClearError,
  onRetry,
  inputValue,
  onInputChange,
  onSubmitMessage,
  onSelectPrompt,
  voiceEnabled = true,
  voiceState = 'idle',
  onVoiceClick,
  onCancelVoice,
  onAttachDocument,
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom whenever messages, loading, or error state changes
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading, error]);

  return (
    <main className="flex-1 flex flex-col h-full overflow-hidden bg-slate-950 relative">
      {/* Scrollable conversation / welcome body */}
      <div className="flex-1 overflow-y-auto px-4 sm:px-6 py-6 flex flex-col">
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col justify-center">
            <WelcomeScreen onSelectPrompt={onSelectPrompt} />
          </div>
        ) : (
          <div className="w-full max-w-3xl mx-auto space-y-6 flex-1">
            {messages.map((msg) => {
              const isUser = msg.role === 'user';
              const timeStr = formatMessageTime(msg.timestamp);

              return (
                <div
                  key={msg.id}
                  className={`flex items-start gap-3 ${
                    isUser ? 'justify-end' : 'justify-start'
                  }`}
                >
                  {/* Assistant Avatar */}
                  {!isUser && (
                    <div
                      className="w-8 h-8 rounded-xl bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 flex items-center justify-center shrink-0 shadow-sm"
                      aria-label="AURA Assistant"
                    >
                      <Sparkles className="w-4 h-4" />
                    </div>
                  )}

                  {/* Message Bubble Container */}
                  <div
                    className={`flex flex-col ${
                      isUser ? 'items-end max-w-[85%] sm:max-w-[75%]' : 'items-start max-w-[92%] sm:max-w-[85%]'
                    }`}
                  >
                    <div
                      className={`rounded-2xl px-4 py-3 text-sm sm:text-base leading-relaxed break-words shadow-sm ${
                        isUser
                          ? 'bg-cyan-600 text-white rounded-tr-sm'
                          : 'bg-slate-900 border border-slate-800 text-slate-100 rounded-tl-sm w-full space-y-3'
                      }`}
                    >
                      {/* Agent Activity Steps for Assistant */}
                      {!isUser && msg.activities && msg.activities.length > 0 && (
                        <div className="mb-2">
                          <AgentActivity activities={msg.activities} />
                        </div>
                      )}

                      {/* Content rendering */}
                      {isUser ? (
                        <div className="whitespace-pre-wrap">{msg.content}</div>
                      ) : (
                        <MarkdownRenderer content={msg.content} />
                      )}
                    </div>

                    {/* Timestamp */}
                    {timeStr && (
                      <span className="text-[10px] text-slate-400 px-1 mt-1 select-none">
                        {timeStr}
                      </span>
                    )}
                  </div>

                  {/* User Avatar */}
                  {isUser && (
                    <div
                      className="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 flex items-center justify-center shrink-0"
                      aria-label="User"
                    >
                      <User className="w-4 h-4" />
                    </div>
                  )}
                </div>
              );
            })}

            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex items-start gap-3 justify-start animate-fade-in">
                <div className="w-8 h-8 rounded-xl bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 flex items-center justify-center shrink-0 shadow-sm">
                  <Sparkles className="w-4 h-4 animate-pulse" />
                </div>
                <div className="bg-slate-900 border border-slate-800 rounded-2xl rounded-tl-sm px-4 py-3 flex items-center gap-3 shadow-sm">
                  <span className="text-xs sm:text-sm text-cyan-400 font-medium flex items-center gap-1.5">
                    <span aria-hidden="true">🧠</span>
                    <span>AURA is working...</span>
                  </span>
                  <div className="flex items-center gap-1 pt-0.5" aria-label="Loading dots">
                    <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                    <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                    <span className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-bounce"></span>
                  </div>
                </div>
              </div>
            )}

            {/* Error Message Banner */}
            {error && (
              <div
                role="alert"
                className="flex items-center gap-3 p-3.5 rounded-xl bg-rose-950/60 border border-rose-800/80 text-rose-200 text-xs sm:text-sm shadow-sm animate-fade-in"
              >
                <AlertCircle className="w-5 h-5 text-rose-400 shrink-0" />
                <div className="flex-1 leading-snug">{error}</div>
                {onRetry && (
                  <button
                    type="button"
                    onClick={onRetry}
                    className="flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-lg bg-rose-900/60 hover:bg-rose-800 border border-rose-700/60 transition-colors text-rose-100 cursor-pointer"
                    title="Retry last message"
                    aria-label="Retry message"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                    <span>Retry</span>
                  </button>
                )}
                {onClearError && (
                  <button
                    type="button"
                    onClick={onClearError}
                    className="text-xs text-rose-400 hover:text-rose-200 p-1 cursor-pointer"
                    title="Dismiss"
                    aria-label="Dismiss error"
                  >
                    ✕
                  </button>
                )}
              </div>
            )}

            {/* Scroll anchor */}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Pinned bottom composer */}
      <div className="shrink-0 bg-gradient-to-t from-slate-950 via-slate-950/95 to-transparent pt-2">
        <MessageInput
          value={inputValue}
          onChange={onInputChange}
          onSubmit={onSubmitMessage}
          disabled={isLoading}
          voiceEnabled={voiceEnabled}
          voiceState={voiceState}
          onVoiceClick={onVoiceClick}
          onCancelVoice={onCancelVoice}
          onAttachDocument={onAttachDocument}
        />
      </div>
    </main>
  );
};

export default ChatArea;
