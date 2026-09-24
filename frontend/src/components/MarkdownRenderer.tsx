import React, { useState } from 'react';
import { Check, Copy } from 'lucide-react';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

interface CodeBlockProps {
  language: string;
  code: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ language, code }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.warn('Failed to copy text:', err);
    }
  };

  return (
    <div className="my-3 rounded-xl overflow-hidden border border-slate-700/80 bg-slate-950/90 shadow-sm">
      <div className="flex items-center justify-between px-3.5 py-1.5 bg-slate-900 border-b border-slate-800 text-xs text-slate-400">
        <span className="font-mono text-[11px] uppercase tracking-wider text-cyan-400 font-semibold">
          {language || 'code'}
        </span>
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-800/80 transition-colors focus:outline-none focus:ring-1 focus:ring-cyan-500/50"
          aria-label={copied ? 'Code copied' : 'Copy code to clipboard'}
        >
          {copied ? (
            <>
              <Check className="w-3.5 h-3.5 text-emerald-400 stroke-[2.5]" />
              <span className="text-[11px] text-emerald-400 font-medium">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3.5 h-3.5" />
              <span className="text-[11px]">Copy</span>
            </>
          )}
        </button>
      </div>
      <pre className="p-3.5 overflow-x-auto text-xs sm:text-sm font-mono text-slate-200 leading-relaxed selection:bg-cyan-900">
        <code>{code}</code>
      </pre>
    </div>
  );
};

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  if (!content) return null;

  // Simple token parser for code blocks and text segments
  const parts: React.ReactNode[] = [];
  const lines = content.split('\n');

  let inCodeBlock = false;
  let codeBlockLang = '';
  let codeBlockBuffer: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (line.startsWith('```')) {
      if (inCodeBlock) {
        // End of code block
        const blockCode = codeBlockBuffer.join('\n');
        parts.push(
          <CodeBlock
            key={`code-block-${parts.length}`}
            language={codeBlockLang}
            code={blockCode}
          />
        );
        inCodeBlock = false;
        codeBlockLang = '';
        codeBlockBuffer = [];
      } else {
        // Start of code block
        inCodeBlock = true;
        codeBlockLang = line.replace('```', '').trim();
        codeBlockBuffer = [];
      }
      continue;
    }

    if (inCodeBlock) {
      codeBlockBuffer.push(line);
      continue;
    }

    // Render headings
    if (line.startsWith('### ')) {
      parts.push(
        <h4 key={`h4-${i}`} className="text-sm sm:text-base font-semibold text-slate-100 mt-3 mb-1">
          {renderInline(line.slice(4))}
        </h4>
      );
      continue;
    }
    if (line.startsWith('## ')) {
      parts.push(
        <h3 key={`h3-${i}`} className="text-base sm:text-lg font-bold text-slate-100 mt-3.5 mb-1.5">
          {renderInline(line.slice(3))}
        </h3>
      );
      continue;
    }
    if (line.startsWith('# ')) {
      parts.push(
        <h2 key={`h2-${i}`} className="text-lg sm:text-xl font-bold text-slate-100 mt-4 mb-2">
          {renderInline(line.slice(2))}
        </h2>
      );
      continue;
    }

    // Blockquote
    if (line.startsWith('> ')) {
      parts.push(
        <blockquote
          key={`quote-${i}`}
          className="border-l-2 border-cyan-500/70 pl-3 py-1 my-2 text-slate-300 italic text-xs sm:text-sm bg-cyan-950/20 rounded-r-md"
        >
          {renderInline(line.slice(2))}
        </blockquote>
      );
      continue;
    }

    // Unordered List item
    if (/^[-*•]\s+/.test(line)) {
      const itemText = line.replace(/^[-*•]\s+/, '');
      parts.push(
        <li key={`li-${i}`} className="flex items-start gap-2 text-sm sm:text-base leading-relaxed my-0.5">
          <span className="text-cyan-400 mt-1.5 select-none text-[8px]">•</span>
          <span className="flex-1">{renderInline(itemText)}</span>
        </li>
      );
      continue;
    }

    // Numbered list item
    const numMatch = line.match(/^(\d+)\.\s+(.*)/);
    if (numMatch) {
      parts.push(
        <li key={`ol-li-${i}`} className="flex items-start gap-2 text-sm sm:text-base leading-relaxed my-0.5">
          <span className="text-cyan-400 font-mono text-xs font-semibold select-none pt-0.5">
            {numMatch[1]}.
          </span>
          <span className="flex-1">{renderInline(numMatch[2])}</span>
        </li>
      );
      continue;
    }

    // Empty line / paragraph spacer
    if (line.trim() === '') {
      parts.push(<div key={`spacer-${i}`} className="h-2" />);
      continue;
    }

    // Regular text line
    parts.push(
      <p key={`p-${i}`} className="text-sm sm:text-base leading-relaxed my-1">
        {renderInline(line)}
      </p>
    );
  }

  // Handle unclosed code block if any
  if (inCodeBlock && codeBlockBuffer.length > 0) {
    parts.push(
      <CodeBlock
        key={`code-block-unclosed`}
        language={codeBlockLang}
        code={codeBlockBuffer.join('\n')}
      />
    );
  }

  return <div className={`space-y-0.5 text-slate-100 ${className}`}>{parts}</div>;
};

/**
 * Parses inline formatting: **bold**, *italic*, and `code`
 */
function renderInline(text: string): React.ReactNode {
  if (!text) return null;

  // Split by inline code: `code`
  const segments = text.split(/(`[^`]+`)/g);

  return segments.map((segment, idx) => {
    if (segment.startsWith('`') && segment.endsWith('`') && segment.length >= 2) {
      return (
        <code
          key={`code-${idx}`}
          className="px-1.5 py-0.5 mx-0.5 rounded-md bg-slate-800 text-cyan-300 font-mono text-xs sm:text-sm border border-slate-700/60"
        >
          {segment.slice(1, -1)}
        </code>
      );
    }

    // Process bold (**text**) and italic (*text*)
    return renderBoldItalic(segment, idx);
  });
}

function renderBoldItalic(text: string, baseKey: number): React.ReactNode {
  // Split by bold (**text**)
  const parts = text.split(/(\*\*[^*]+\*\*)/g);

  return parts.map((part, pIdx) => {
    if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
      return (
        <strong key={`bold-${baseKey}-${pIdx}`} className="font-semibold text-white">
          {part.slice(2, -2)}
        </strong>
      );
    }

    // Split by italic (*text*)
    const subParts = part.split(/(\*[^*]+\*)/g);
    return subParts.map((sub, sIdx) => {
      if (sub.startsWith('*') && sub.endsWith('*') && sub.length >= 2) {
        return (
          <em key={`em-${baseKey}-${pIdx}-${sIdx}`} className="italic text-slate-200">
            {sub.slice(1, -1)}
          </em>
        );
      }
      return sub;
    });
  });
}

export default MarkdownRenderer;
