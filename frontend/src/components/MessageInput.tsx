import React, { FormEvent, KeyboardEvent, useRef, useEffect } from 'react';
import { Mic, Send, Square, Loader2, Volume2, AlertTriangle, X, Paperclip } from 'lucide-react';

export type VoiceState = 'idle' | 'requesting' | 'listening' | 'processing' | 'speaking' | 'error';

interface MessageInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: (value: string) => void;
  disabled?: boolean;
  voiceEnabled?: boolean;
  voiceState?: VoiceState;
  onVoiceClick?: () => void;
  onCancelVoice?: () => void;
  onAttachDocument?: () => void;
}

export const MessageInput: React.FC<MessageInputProps> = ({
  value,
  onChange,
  onSubmit,
  disabled = false,
  voiceEnabled = true,
  voiceState = 'idle',
  onVoiceClick,
  onCancelVoice,
  onAttachDocument,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto-adjust textarea height on input changes
  useEffect(() => {
    const el = textareaRef.current;
    if (el) {
      el.style.height = 'auto';
      const newHeight = Math.min(Math.max(el.scrollHeight, 40), 160);
      el.style.height = `${newHeight}px`;
    }
  }, [value]);

  const handleSubmit = (e?: FormEvent) => {
    if (e) e.preventDefault();
    if (!value.trim() || disabled || voiceState === 'listening' || voiceState === 'processing') return;
    onSubmit(value);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    } else if (e.key === 'Escape' && voiceState !== 'idle') {
      onCancelVoice?.();
    }
  };

  // State-aware placeholder text
  let placeholderText = 'Message AURA or ask a question... (Enter to send, Shift+Enter for new line)';
  if (voiceState === 'listening') {
    placeholderText = 'Listening... Speak into your microphone...';
  } else if (voiceState === 'processing') {
    placeholderText = 'Processing speech and formulating answer...';
  } else if (voiceState === 'requesting') {
    placeholderText = 'Requesting microphone permission...';
  } else if (voiceState === 'speaking') {
    placeholderText = 'AURA is speaking response...';
  } else if (disabled) {
    placeholderText = 'AURA is responding...';
  }

  const isInputLocked = disabled || voiceState === 'listening' || voiceState === 'processing';
  const isSendDisabled = !value.trim() || isInputLocked;

  return (
    <div className="w-full max-w-3xl mx-auto px-4 pb-4 sm:pb-6">
      {/* Voice Status Pill / Banner (Active when not idle) */}
      {voiceState !== 'idle' && (
        <div
          role="status"
          aria-live="polite"
          className="mb-2 flex items-center justify-between px-3.5 py-1.5 rounded-xl text-xs sm:text-sm font-medium transition-all shadow-sm animate-fade-in border"
          style={{
            borderColor:
              voiceState === 'listening'
                ? 'rgba(244, 63, 94, 0.4)'
                : voiceState === 'processing'
                ? 'rgba(6, 182, 212, 0.4)'
                : voiceState === 'speaking'
                ? 'rgba(168, 85, 247, 0.4)'
                : voiceState === 'requesting'
                ? 'rgba(245, 158, 11, 0.4)'
                : 'rgba(239, 68, 68, 0.4)',
            backgroundColor:
              voiceState === 'listening'
                ? 'rgba(76, 5, 25, 0.85)'
                : voiceState === 'processing'
                ? 'rgba(8, 51, 68, 0.85)'
                : voiceState === 'speaking'
                ? 'rgba(59, 7, 100, 0.85)'
                : voiceState === 'requesting'
                ? 'rgba(69, 26, 3, 0.85)'
                : 'rgba(69, 10, 10, 0.85)',
            color:
              voiceState === 'listening'
                ? '#fecdd3'
                : voiceState === 'processing'
                ? '#a5f3fc'
                : voiceState === 'speaking'
                ? '#e9d5ff'
                : voiceState === 'requesting'
                ? '#fde68a'
                : '#fca5a5',
          }}
        >
          <div className="flex items-center gap-2">
            {voiceState === 'listening' && (
              <>
                <span className="relative flex h-2.5 w-2.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-rose-500"></span>
                </span>
                <span>Listening... Speak now, click Stop when finished.</span>
              </>
            )}
            {voiceState === 'processing' && (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                <span>Processing... Transcribing speech &amp; thinking...</span>
              </>
            )}
            {voiceState === 'requesting' && (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin text-amber-400" />
                <span>Requesting permission... Please allow microphone access.</span>
              </>
            )}
            {voiceState === 'speaking' && (
              <>
                <Volume2 className="w-3.5 h-3.5 animate-pulse text-purple-300" />
                <span>Speaking... AURA voice response is playing.</span>
              </>
            )}
            {voiceState === 'error' && (
              <>
                <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
                <span>Voice input error. Click mic icon to retry.</span>
              </>
            )}
          </div>

          {/* Quick action button inside pill */}
          <div className="flex items-center gap-1.5">
            {voiceState === 'listening' && (
              <button
                type="button"
                onClick={onVoiceClick}
                className="px-2.5 py-0.5 rounded-lg bg-rose-800/80 hover:bg-rose-700 text-[11px] font-semibold text-rose-100 transition-colors focus:outline-none focus:ring-1 focus:ring-rose-400"
                title="Stop recording"
                aria-label="Stop recording"
              >
                Stop
              </button>
            )}
            {voiceState === 'speaking' && (
              <button
                type="button"
                onClick={onVoiceClick}
                className="px-2.5 py-0.5 rounded-lg bg-purple-800/80 hover:bg-purple-700 text-[11px] font-semibold text-purple-100 transition-colors focus:outline-none focus:ring-1 focus:ring-purple-400"
                title="Stop audio playback"
                aria-label="Stop audio playback"
              >
                Stop Audio
              </button>
            )}
            {onCancelVoice && (
              <button
                type="button"
                onClick={onCancelVoice}
                className="p-1 rounded-md hover:bg-white/10 opacity-70 hover:opacity-100 transition-opacity focus:outline-none focus:ring-1 focus:ring-white/40"
                title="Dismiss (Esc)"
                aria-label="Dismiss voice status"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Composer Input Box */}
      <form
        onSubmit={handleSubmit}
        className="relative flex items-end gap-2 bg-slate-900 border border-slate-800 focus-within:border-cyan-500/50 focus-within:ring-1 focus-within:ring-cyan-500/40 rounded-2xl p-2 sm:p-2.5 transition-all shadow-lg shadow-black/25"
      >
        {/* Document Attachment Button */}
        {onAttachDocument && (
          <button
            type="button"
            onClick={onAttachDocument}
            disabled={isInputLocked}
            title="Attach document to knowledge base (PDF, DOCX, TXT)"
            aria-label="Attach document"
            className="p-2 rounded-xl text-slate-400 hover:text-cyan-400 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-cyan-500/40 transition-colors disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer shrink-0"
          >
            <Paperclip className="w-4 h-4 sm:w-5 sm:h-5" />
          </button>
        )}

        {/* Multiline auto-adjusting textarea */}
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholderText}
          disabled={isInputLocked}
          aria-label="Message input"
          className="flex-1 max-h-40 min-h-[38px] py-2 px-2 bg-transparent text-sm sm:text-base text-slate-100 placeholder-slate-500 resize-none focus:outline-none leading-relaxed disabled:opacity-60 disabled:cursor-not-allowed"
        />

        {/* Right action group */}
        <div className="flex items-center gap-1.5 shrink-0">
          {/* Microphone button with state awareness */}
          {voiceEnabled && (
            <button
              type="button"
              onClick={onVoiceClick}
              disabled={voiceState === 'processing'}
              title={
                voiceState === 'idle'
                  ? 'Click to speak'
                  : voiceState === 'requesting'
                  ? 'Requesting microphone...'
                  : voiceState === 'listening'
                  ? 'Listening... (Click to stop)'
                  : voiceState === 'processing'
                  ? 'Processing audio...'
                  : voiceState === 'speaking'
                  ? 'Speaking... (Click to stop)'
                  : 'Voice error (Click to retry)'
              }
              aria-label={
                voiceState === 'idle'
                  ? 'Start voice input'
                  : voiceState === 'listening'
                  ? 'Stop voice recording'
                  : voiceState === 'speaking'
                  ? 'Stop audio playback'
                  : 'Microphone'
              }
              className={`p-2 rounded-xl transition-all focus:outline-none focus:ring-2 cursor-pointer disabled:cursor-not-allowed ${
                voiceState === 'listening'
                  ? 'bg-rose-950/90 border border-rose-500/60 text-rose-300 hover:bg-rose-900 focus:ring-rose-500/50'
                  : voiceState === 'processing'
                  ? 'bg-cyan-950/90 border border-cyan-500/50 text-cyan-300 focus:ring-cyan-500/50'
                  : voiceState === 'speaking'
                  ? 'bg-purple-950/90 border border-purple-500/60 text-purple-300 hover:bg-purple-900 focus:ring-purple-500/50'
                  : voiceState === 'requesting'
                  ? 'bg-amber-950/90 border border-amber-500/50 text-amber-300 focus:ring-amber-500/50'
                  : voiceState === 'error'
                  ? 'text-rose-400 hover:text-rose-200 hover:bg-rose-950/60 focus:ring-rose-500/50'
                  : 'text-slate-400 hover:text-cyan-400 hover:bg-slate-800 focus:ring-slate-700'
              }`}
            >
              {voiceState === 'idle' && <Mic className="w-4 h-4 sm:w-5 sm:h-5" />}
              {voiceState === 'requesting' && (
                <Loader2 className="w-4 h-4 sm:w-5 sm:h-5 animate-spin text-amber-400" />
              )}
              {voiceState === 'listening' && (
                <Square className="w-4 h-4 sm:w-5 sm:h-5 text-rose-400 fill-rose-400" />
              )}
              {voiceState === 'processing' && (
                <Loader2 className="w-4 h-4 sm:w-5 sm:h-5 animate-spin text-cyan-400" />
              )}
              {voiceState === 'speaking' && (
                <Volume2 className="w-4 h-4 sm:w-5 sm:h-5 text-purple-400 animate-pulse" />
              )}
              {voiceState === 'error' && (
                <AlertTriangle className="w-4 h-4 sm:w-5 sm:h-5 text-rose-400" />
              )}
            </button>
          )}

          {/* Send button */}
          <button
            type="submit"
            disabled={isSendDisabled}
            title="Send message (Enter)"
            aria-label="Send message"
            className="p-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-800 disabled:text-slate-600 text-white transition-colors shadow-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/50 cursor-pointer disabled:cursor-not-allowed"
          >
            <Send className="w-4 h-4 sm:w-5 sm:h-5" />
          </button>
        </div>
      </form>

      {/* Helpful keyboard hint */}
      <div className="mt-2 flex items-center justify-between px-1 text-[11px] text-slate-400 select-none">
        <span>Use <kbd className="px-1 py-0.5 rounded bg-slate-800 border border-slate-700 font-mono text-[10px] text-slate-300">Shift</kbd> + <kbd className="px-1 py-0.5 rounded bg-slate-800 border border-slate-700 font-mono text-[10px] text-slate-300">Enter</kbd> for new line</span>
        <span>AURA AI Assistant</span>
      </div>
    </div>
  );
};

export default MessageInput;
