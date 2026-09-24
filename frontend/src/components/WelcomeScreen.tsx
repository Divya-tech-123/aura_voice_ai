import React from 'react';
import { Sparkles, Brain, Calculator, FileSearch, CloudSun, ArrowRight } from 'lucide-react';

interface WelcomeScreenProps {
  onSelectPrompt: (prompt: string) => void;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onSelectPrompt }) => {
  const examplePrompts = [
    {
      title: 'Explain machine learning',
      description: 'Understand core AI concepts & architecture',
      category: 'Concept & Theory',
      icon: Brain,
      prompt: 'Explain machine learning concepts and neural networks.',
    },
    {
      title: 'Calculate 125 × 32',
      description: 'Execute arithmetic and multi-step calculations',
      category: 'Tools & Math',
      icon: Calculator,
      prompt: 'Calculate 125 × 32',
    },
    {
      title: 'Search my documents',
      description: 'Query indexed PDF and TXT knowledge base',
      category: 'RAG & Knowledge',
      icon: FileSearch,
      prompt: 'Search my documents for key summary points.',
    },
    {
      title: 'Check weather conditions',
      description: 'Look up live weather using real-time tool',
      category: 'Real-time Tools',
      icon: CloudSun,
      prompt: 'What is the weather in Tokyo?',
    },
  ];

  return (
    <div className="flex flex-col items-center justify-center flex-1 max-w-3xl w-full mx-auto px-4 py-8 text-center select-none">
      {/* Brand Icon Emblem */}
      <div className="relative mb-5">
        <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-2xl bg-cyan-950/40 border border-cyan-500/30 flex items-center justify-center shadow-lg shadow-cyan-950/50">
          <Sparkles className="w-8 h-8 sm:w-10 sm:h-10 text-cyan-400" />
        </div>
      </div>

      {/* Main Titles */}
      <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-slate-100">
        AURA
      </h1>
      <h2 className="mt-2 text-base sm:text-lg font-medium text-cyan-400">
        Your AI Unified Response Assistant
      </h2>
      <p className="mt-3 text-sm sm:text-base text-slate-400 max-w-lg leading-relaxed">
        Ask questions, use tools, search your documents, or talk to AURA.
      </p>

      {/* Example Prompt Grid */}
      <div className="w-full grid grid-cols-1 sm:grid-cols-2 gap-3.5 mt-8 text-left">
        {examplePrompts.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.title}
              type="button"
              onClick={() => onSelectPrompt(item.prompt)}
              className="group flex flex-col justify-between p-4 rounded-xl bg-slate-900/80 hover:bg-slate-850 border border-slate-800 hover:border-cyan-500/40 transition-all duration-200 text-left shadow-sm focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
            >
              <div>
                <div className="flex items-center justify-between mb-2.5">
                  <div className="p-2 rounded-lg bg-slate-800 border border-slate-700/60 text-cyan-400 group-hover:text-cyan-300 transition-colors">
                    <Icon className="w-4 h-4" />
                  </div>
                  <span className="text-[11px] font-medium text-slate-400 group-hover:text-slate-300 transition-colors">
                    {item.category}
                  </span>
                </div>
                <div className="text-sm font-semibold text-slate-200 group-hover:text-white transition-colors">
                  &ldquo;{item.title}&rdquo;
                </div>
                <p className="text-xs text-slate-400 mt-1 leading-snug">
                  {item.description}
                </p>
              </div>

              <div className="flex items-center gap-1.5 mt-3.5 text-xs font-medium text-cyan-400/90 group-hover:text-cyan-300 transition-colors">
                <span>Use prompt</span>
                <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default WelcomeScreen;
