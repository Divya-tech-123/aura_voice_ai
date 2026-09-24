import React, { useState } from 'react';
import {
  ChevronDown,
  Check,
  AlertTriangle,
  Loader2,
  Brain,
  Calculator,
  Database,
  Search,
  CloudSun,
  FileText,
  MessageSquare,
  Sparkles,
} from 'lucide-react';

export interface ActivityItem {
  type: string;
  label: string;
  status: 'completed' | 'failed' | 'in_progress' | 'pending' | string;
}

export interface AgentActivityProps {
  activities?: ActivityItem[];
  defaultExpanded?: boolean;
  className?: string;
}

/**
 * Return an appropriate Lucide icon for a given activity type or label.
 */
function getActivityTypeIcon(type: string, label: string): React.ReactElement {
  const cleanType = (type || '').toLowerCase();
  const cleanLabel = (label || '').toLowerCase();

  if (cleanType.includes('plan') || cleanLabel.includes('plan')) {
    return <Brain className="w-3.5 h-3.5 text-indigo-400" />;
  }
  if (cleanType.includes('calc') || cleanLabel.includes('calc')) {
    return <Calculator className="w-3.5 h-3.5 text-emerald-400" />;
  }
  if (
    cleanType.includes('retriev') ||
    cleanType.includes('rag') ||
    cleanLabel.includes('retriev') ||
    cleanLabel.includes('document')
  ) {
    return <Database className="w-3.5 h-3.5 text-cyan-400" />;
  }
  if (cleanType.includes('search') || cleanLabel.includes('search')) {
    return <Search className="w-3.5 h-3.5 text-amber-400" />;
  }
  if (cleanType.includes('weather') || cleanLabel.includes('weather')) {
    return <CloudSun className="w-3.5 h-3.5 text-cyan-400" />;
  }
  if (cleanType.includes('file') || cleanLabel.includes('file')) {
    return <FileText className="w-3.5 h-3.5 text-purple-400" />;
  }
  if (cleanType.includes('respond') || cleanType.includes('response') || cleanLabel.includes('response')) {
    return <MessageSquare className="w-3.5 h-3.5 text-teal-400" />;
  }

  return <Sparkles className="w-3.5 h-3.5 text-cyan-400" />;
}

/**
 * Return a user-safe label for an activity item.
 * Guarantees that internal prompts, raw exceptions, or secrets are never shown.
 */
function getSafeLabel(item: ActivityItem): string {
  const raw = (item.label || '').trim();
  const lower = raw.toLowerCase();

  if (lower.includes('plan')) return 'Planning task';
  if (lower.includes('calc')) return 'Using Calculator';
  if (lower.includes('retriev') || lower.includes('document')) return 'Retrieving documents';
  if (lower.includes('weather')) return 'Using Weather tool';
  if (lower.includes('search')) return 'Searching knowledge base';
  if (lower.includes('file')) return 'Reading file';
  if (lower.includes('generat') || lower.includes('respond') || lower.includes('response')) {
    return 'Generating response';
  }

  // Fallback to sanitized raw label (without leaking tracebacks or secret patterns)
  if (raw.length > 50 || raw.includes('Traceback') || raw.includes('Error') || raw.includes('=')) {
    return 'Executing action';
  }

  return raw || 'Processing step';
}

/**
 * AgentActivity Component
 *
 * Renders a compact, expandable panel detailing high-level agent execution steps.
 * Designed to be subtle, accessible, and user-safe.
 */
export const AgentActivity: React.FC<AgentActivityProps> = ({
  activities,
  defaultExpanded,
  className = '',
}) => {
  if (!activities || activities.length === 0) {
    return null;
  }

  const hasFailed = activities.some((a) => a.status === 'failed');
  const [isExpanded, setIsExpanded] = useState<boolean>(defaultExpanded ?? hasFailed);

  const completedCount = activities.filter((a) => a.status === 'completed').length;
  const totalCount = activities.length;

  return (
    <section
      aria-label="AURA execution activities"
      className={`rounded-xl border border-slate-800 bg-slate-950/70 p-2.5 text-xs text-slate-300 shadow-sm transition-all duration-200 ${className}`}
    >
      {/* Expand/Collapse Toggle Header */}
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
        aria-controls="aura-activity-steps"
        className="w-full flex items-center justify-between gap-2 px-1 py-0.5 text-left rounded-lg hover:bg-slate-900/60 focus:outline-none focus:ring-1 focus:ring-cyan-500/50 transition-colors"
      >
        <div className="flex items-center gap-2 font-medium text-slate-200">
          <Sparkles className="w-3.5 h-3.5 text-cyan-400 shrink-0" />
          <span>AURA activity</span>
          <span className="text-[10px] text-slate-400 bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded-md font-mono">
            {completedCount}/{totalCount}
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
          <span>{isExpanded ? 'Hide' : 'Show'}</span>
          <ChevronDown
            className={`w-3.5 h-3.5 text-slate-400 transition-transform duration-200 ${
              isExpanded ? 'rotate-180' : ''
            }`}
          />
        </div>
      </button>

      {/* Expandable Activity List */}
      {isExpanded && (
        <div
          id="aura-activity-steps"
          role="list"
          className="mt-2.5 pt-2 border-t border-slate-800/80 space-y-2 animate-fade-in"
        >
          {activities.map((item, idx) => {
            const isCompleted = item.status === 'completed';
            const isFailed = item.status === 'failed';
            const isInProgress = item.status === 'in_progress';
            const labelText = getSafeLabel(item);

            return (
              <div
                key={`activity-${idx}-${item.type}`}
                role="listitem"
                className="flex items-start gap-2.5 py-0.5"
              >
                {/* Status Indicator Icon */}
                <div className="mt-0.5 shrink-0 flex items-center justify-center">
                  {isCompleted && (
                    <div
                      className="w-4 h-4 rounded-full bg-emerald-950/80 border border-emerald-700/60 flex items-center justify-center text-emerald-400"
                      title="Step completed"
                    >
                      <Check className="w-2.5 h-2.5 stroke-[3]" />
                    </div>
                  )}
                  {isFailed && (
                    <div
                      className="w-4 h-4 rounded-full bg-rose-950/80 border border-rose-700/60 flex items-center justify-center text-rose-400"
                      title="Step failed"
                    >
                      <AlertTriangle className="w-2.5 h-2.5 stroke-[2.5]" />
                    </div>
                  )}
                  {isInProgress && (
                    <div
                      className="w-4 h-4 rounded-full bg-cyan-950/80 border border-cyan-700/60 flex items-center justify-center text-cyan-400"
                      title="In progress"
                    >
                      <Loader2 className="w-2.5 h-2.5 animate-spin" />
                    </div>
                  )}
                  {!isCompleted && !isFailed && !isInProgress && (
                    <div className="w-4 h-4 rounded-full bg-slate-900 border border-slate-700 flex items-center justify-center">
                      <span className="w-1.5 h-1.5 rounded-full bg-slate-500" />
                    </div>
                  )}
                </div>

                {/* Activity Content */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1.5 leading-snug">
                    <span className="shrink-0">{getActivityTypeIcon(item.type, labelText)}</span>
                    <span
                      className={`truncate ${
                        isFailed
                          ? 'text-rose-300 font-medium'
                          : isCompleted
                          ? 'text-slate-200'
                          : 'text-slate-400'
                      }`}
                    >
                      {labelText}
                    </span>
                  </div>

                  {/* Safe Error State */}
                  {isFailed && (
                    <div className="mt-1 flex items-center gap-1.5 text-[11px] text-amber-300 bg-amber-950/40 border border-amber-800/40 rounded px-2 py-0.5">
                      <span>⚠️ AURA couldn't complete this step.</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
};

export default AgentActivity;
