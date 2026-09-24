import { useState, useRef, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { ChatArea } from './components/ChatArea';
import { SettingsModal } from './components/SettingsModal';
import { VoiceState } from './components/MessageInput';
import { sendChatMessage, sendVoiceAudio } from './services/api';
import { AudioRecorder } from './services/audioRecorder';
import { AppSettings, ChatMessage, ConversationSession, ThemePreference } from './types';

const SETTINGS_STORAGE_KEY = 'aura_settings';
const CONVERSATIONS_STORAGE_KEY = 'aura_conversations';

const DEFAULT_SETTINGS: AppSettings = {
  theme: 'dark',
  voiceEnabled: true,
  autoPlayVoice: true,
};

function loadStoredSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
    if (raw) {
      return { ...DEFAULT_SETTINGS, ...JSON.parse(raw) };
    }
  } catch (e) {
    console.warn('Could not read settings from localStorage:', e);
  }
  return DEFAULT_SETTINGS;
}

function loadStoredConversations(): ConversationSession[] {
  try {
    const raw = localStorage.getItem(CONVERSATIONS_STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        return parsed;
      }
    }
  } catch (e) {
    console.warn('Could not read conversations from localStorage:', e);
  }
  return [];
}

export function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [settings, setSettings] = useState<AppSettings>(loadStoredSettings);

  const [conversations, setConversations] = useState<ConversationSession[]>(loadStoredConversations);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);

  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastMessageText, setLastMessageText] = useState<string | null>(null);
  const [voiceState, setVoiceState] = useState<VoiceState>('idle');

  const audioRecorderRef = useRef<AudioRecorder | null>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const documentUploadTriggerRef = useRef<(() => void) | null>(null);

  // Apply Theme Preference to document element
  useEffect(() => {
    const root = document.documentElement;
    const applyTheme = (theme: ThemePreference) => {
      if (theme === 'dark') {
        root.classList.add('dark');
        root.classList.remove('light');
      } else if (theme === 'light') {
        root.classList.remove('dark');
        root.classList.add('light');
      } else {
        // System preference
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (prefersDark) {
          root.classList.add('dark');
          root.classList.remove('light');
        } else {
          root.classList.remove('dark');
          root.classList.add('light');
        }
      }
    };

    applyTheme(settings.theme);

    // If system theme, listen to changes
    if (settings.theme === 'system') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      const listener = () => applyTheme('system');
      mediaQuery.addEventListener('change', listener);
      return () => mediaQuery.removeEventListener('change', listener);
    }
  }, [settings.theme]);

  // Persist settings changes
  const handleUpdateSettings = (newSettings: Partial<AppSettings>) => {
    setSettings((prev) => {
      const updated = { ...prev, ...newSettings };
      try {
        localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(updated));
      } catch (err) {
        console.warn('Failed to save settings:', err);
      }
      return updated;
    });
  };

  // Persist conversations helper
  const saveConversations = (convList: ConversationSession[]) => {
    setConversations(convList);
    try {
      localStorage.setItem(CONVERSATIONS_STORAGE_KEY, JSON.stringify(convList));
    } catch (err) {
      console.warn('Failed to save conversations:', err);
    }
  };

  useEffect(() => {
    audioRecorderRef.current = new AudioRecorder();
    return () => {
      if (audioRecorderRef.current) {
        audioRecorderRef.current.cancel();
      }
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
    };
  }, []);

  const handleAttachDocument = () => {
    documentUploadTriggerRef.current?.();
  };

  const handleToggleSidebar = () => {
    setIsSidebarOpen((prev) => !prev);
  };

  const handleCloseSidebar = () => {
    setIsSidebarOpen(false);
  };

  const handleNewChat = () => {
    if (audioRecorderRef.current) {
      audioRecorderRef.current.cancel();
    }
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    setVoiceState('idle');
    setMessages([]);
    setActiveConversationId(null);
    setInputMessage('');
    setError(null);
    setLastMessageText(null);
  };

  const handleSelectConversation = (id: string) => {
    const target = conversations.find((c) => c.id === id);
    if (target) {
      setActiveConversationId(target.id);
      setMessages(target.messages || []);
      setError(null);
      setLastMessageText(null);
    }
  };

  const handleDeleteConversation = (id: string) => {
    const remaining = conversations.filter((c) => c.id !== id);
    saveConversations(remaining);
    if (activeConversationId === id) {
      handleNewChat();
    }
  };

  const handleClearActiveConversation = () => {
    if (!activeConversationId) {
      setMessages([]);
      return;
    }
    const remaining = conversations.filter((c) => c.id !== activeConversationId);
    saveConversations(remaining);
    handleNewChat();
  };

  const handleSelectPrompt = (prompt: string) => {
    setInputMessage(prompt);
  };

  // Add messages to active conversation and update state
  const appendMessagesToActiveConversation = (newMsgs: ChatMessage[]) => {
    setMessages((prev) => {
      const updatedMessages = [...prev, ...newMsgs];

      let currentId = activeConversationId;
      let convList = [...conversations];

      if (!currentId) {
        // Create new conversation session
        const firstUserMsg = newMsgs.find((m) => m.role === 'user') || newMsgs[0];
        const title = firstUserMsg
          ? firstUserMsg.content.slice(0, 32).trim() + (firstUserMsg.content.length > 32 ? '...' : '')
          : 'New Conversation';

        currentId = `conv-${Date.now()}`;
        setActiveConversationId(currentId);

        const newConv: ConversationSession = {
          id: currentId,
          title,
          createdAt: Date.now(),
          messages: updatedMessages,
        };
        convList = [newConv, ...convList];
      } else {
        // Update existing conversation
        convList = convList.map((c) =>
          c.id === currentId ? { ...c, messages: updatedMessages } : c
        );
      }

      saveConversations(convList);
      return updatedMessages;
    });
  };

  const handleSubmitMessage = async (messageText: string) => {
    const trimmed = messageText.trim();
    if (!trimmed || isLoading || voiceState === 'listening' || voiceState === 'processing') return;

    // 1. Add user message
    const userMessage: ChatMessage = {
      id: `user-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
      role: 'user',
      content: trimmed,
      timestamp: new Date().toISOString(),
    };

    appendMessagesToActiveConversation([userMessage]);
    setLastMessageText(trimmed);
    setInputMessage('');
    setError(null);
    setIsLoading(true);

    try {
      // 2. Send message to /api/chat
      const chatResponse = await sendChatMessage(trimmed);

      // 3. Add assistant response
      const assistantMessage: ChatMessage = {
        id: `aura-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        role: 'assistant',
        content: chatResponse.response,
        activities: chatResponse.activities,
        timestamp: new Date().toISOString(),
      };

      appendMessagesToActiveConversation([assistantMessage]);
    } catch (err: any) {
      console.error('Error communicating with AURA API:', err);
      const userFriendlyError =
        err?.message || 'An unexpected error occurred while communicating with AURA.';
      setError(userFriendlyError);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVoiceToggle = async () => {
    if (!settings.voiceEnabled) {
      setError('Voice input is disabled in Settings. Please enable it to use voice.');
      return;
    }

    const recorder = audioRecorderRef.current;
    if (!recorder) return;

    // 1. Stop playback if currently speaking
    if (voiceState === 'speaking') {
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
      setVoiceState('idle');
      return;
    }

    // 2. Stop recording and process
    if (voiceState === 'listening') {
      console.debug('[App] Stopping voice recording and processing audio...');
      setVoiceState('processing');
      try {
        const audioBlob = await recorder.stop();
        console.debug('[App] Voice recording stopped. AudioBlob produced:', {
          size: `${audioBlob.size} bytes`,
          type: audioBlob.type,
        });
        await handleProcessVoiceAudio(audioBlob);
      } catch (err: any) {
        console.warn('[App] Voice recording error:', err);
        const userMsg = err?.message || 'No speech detected. Please speak clearly into your microphone.';
        setError(userMsg);
        setVoiceState('error');
      }
      return;
    }

    if (voiceState === 'processing' || voiceState === 'requesting') {
      return;
    }

    // 3. Request microphone permission & start
    setError(null);
    setVoiceState('requesting');
    console.debug('[App] Requesting microphone access and starting recorder...');

    try {
      await recorder.start();
      console.debug('[App] Microphone listening active. Speak now.');
      setVoiceState('listening');
    } catch (err: any) {
      console.error('[App] Microphone access denied or failed:', err);
      const userMsg =
        err?.message || 'Microphone access was denied. Please allow microphone access in your browser settings.';
      setError(userMsg);
      setVoiceState('error');
    }
  };

  const handleProcessVoiceAudio = async (audioBlob: Blob) => {
    try {
      console.debug('[App] Dispatching voice audio to backend:', { size: audioBlob.size, type: audioBlob.type });
      const result = await sendVoiceAudio(audioBlob);

      const userMsg: ChatMessage = {
        id: `user-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        role: 'user',
        content: result.user_text,
        timestamp: new Date().toISOString(),
      };

      const assistantMsg: ChatMessage = {
        id: `aura-${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        role: 'assistant',
        content: result.response,
        activities: result.activities,
        timestamp: new Date().toISOString(),
      };

      appendMessagesToActiveConversation([userMsg, assistantMsg]);
      setLastMessageText(result.user_text);

      // Auto-play audio if returned and enabled in settings
      if (result.audio && settings.autoPlayVoice) {
        try {
          const audio = new Audio(result.audio);
          currentAudioRef.current = audio;
          setVoiceState('speaking');

          audio.onended = () => {
            currentAudioRef.current = null;
            setVoiceState('idle');
          };

          audio.onerror = (e) => {
            console.warn('Audio playback error:', e);
            currentAudioRef.current = null;
            setVoiceState('idle');
          };

          await audio.play();
        } catch (playErr) {
          console.warn('Audio autoplay failed or blocked:', playErr);
          currentAudioRef.current = null;
          setVoiceState('idle');
        }
      } else {
        setVoiceState('idle');
      }
    } catch (err: any) {
      console.error('Voice processing failed:', err);
      const userFriendlyError =
        err?.message || 'An error occurred while transcribing and processing your voice input.';
      setError(userFriendlyError);
      setVoiceState('error');
    }
  };

  const handleCancelVoice = () => {
    if (audioRecorderRef.current) {
      audioRecorderRef.current.cancel();
    }
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    setVoiceState('idle');
  };

  const handleRetry = () => {
    if (lastMessageText && !isLoading && voiceState === 'idle') {
      setError(null);
      handleSubmitMessage(lastMessageText);
    }
  };

  const handleClearError = () => {
    setError(null);
    if (voiceState === 'error') {
      setVoiceState('idle');
    }
  };

  const headerStatus =
    voiceState === 'listening' || voiceState === 'requesting' || voiceState === 'processing'
      ? 'thinking'
      : voiceState === 'speaking'
      ? 'ready'
      : isLoading
      ? 'thinking'
      : error
      ? 'error'
      : 'ready';

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950 font-sans text-slate-100">
      {/* Navigation Sidebar */}
      <Sidebar
        isOpen={isSidebarOpen}
        onClose={handleCloseSidebar}
        onNewChat={handleNewChat}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelectConversation={handleSelectConversation}
        onDeleteConversation={handleDeleteConversation}
        onOpenSettings={() => setIsSettingsOpen(true)}
        documentUploadTriggerRef={documentUploadTriggerRef}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        <Header
          onToggleSidebar={handleToggleSidebar}
          onOpenSettings={() => setIsSettingsOpen(true)}
          status={headerStatus}
        />
        <ChatArea
          messages={messages}
          isLoading={isLoading}
          error={error}
          onClearError={handleClearError}
          onRetry={lastMessageText ? handleRetry : undefined}
          inputValue={inputMessage}
          onInputChange={setInputMessage}
          onSubmitMessage={handleSubmitMessage}
          onSelectPrompt={handleSelectPrompt}
          voiceEnabled={settings.voiceEnabled}
          voiceState={voiceState}
          onVoiceClick={handleVoiceToggle}
          onCancelVoice={handleCancelVoice}
          onAttachDocument={handleAttachDocument}
        />
      </div>

      {/* Settings Dialog Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onUpdateSettings={handleUpdateSettings}
        onClearConversation={handleClearActiveConversation}
        hasMessages={messages.length > 0}
      />
    </div>
  );
}

export default App;
