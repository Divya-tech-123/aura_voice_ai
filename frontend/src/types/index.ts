import { ActivityItem } from '../components/AgentActivity';

export type ThemePreference = 'dark' | 'light' | 'system';

export interface AppSettings {
  theme: ThemePreference;
  voiceEnabled: boolean;
  autoPlayVoice: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string; // ISO string for easy serialization
  activities?: ActivityItem[];
}

export interface ConversationSession {
  id: string;
  title: string;
  createdAt: number;
  messages: ChatMessage[];
}
