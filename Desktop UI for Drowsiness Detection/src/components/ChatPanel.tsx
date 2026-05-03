import React, { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { Send, Bot, User, Loader2, AlertCircle } from 'lucide-react';
import { apiPost } from '../lib/api';

// ── Types ─────────────────────────────────────────────────────────────────────

interface BotMessage {
  id: string;
  role: 'bot';
  text: string;
  columnNames?: string[];
  rows?: unknown[][];
  chartSuggestion?: string;
  riskLevel?: string;
  loading?: boolean;
  error?: boolean;
}

interface UserMessage {
  id: string;
  role: 'user';
  text: string;
}

type ChatMessage = UserMessage | BotMessage;

// ── Quick question presets ────────────────────────────────────────────────────

const QUICK_QUESTIONS = [
  { label: 'KPI hôm nay',         q: 'Tổng quan hôm nay' },
  { label: 'Phòng rủi ro cao',    q: 'Phòng nào có rủi ro cao nhất hôm nay?' },
  { label: 'So sánh tuần',        q: 'So sánh tuần này với tuần trước theo từng phòng' },
  { label: 'Top học sinh',        q: 'Top 10 học sinh ngủ gật nhiều nhất tháng này' },
  { label: 'Giờ dễ ngủ',         q: 'Khung giờ nào dễ ngủ gật nhất?' },
  { label: 'Đang ngủ gật',       q: 'Ai đang ngủ gật ngay lúc này?' },
];

// ── Sub-components ────────────────────────────────────────────────────────────

function RiskBadge({ level }: { level: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    high:   { label: 'Rủi ro cao', cls: 'bg-red-100 text-red-700' },
    medium: { label: 'Trung bình', cls: 'bg-yellow-100 text-yellow-700' },
    low:    { label: 'Thấp',       cls: 'bg-green-100 text-green-700' },
  };
  const item = map[level];
  if (!item) return null;
  return (
    <span className={`ml-1.5 text-xs px-2 py-0.5 rounded-full font-medium ${item.cls}`}>
      {item.label}
    </span>
  );
}

function ResultTable({ columns, rows }: { columns: string[]; rows: unknown[][] }) {
  if (!columns.length || !rows.length) return null;
  return (
    <div className="mt-2 overflow-x-auto rounded border border-gray-200 text-xs">
      <table className="w-full border-collapse">
        <thead>
          <tr className="bg-gray-100">
            {columns.map((col) => (
              <th
                key={col}
                className="px-2 py-1.5 text-left font-semibold text-gray-600 border-b border-gray-200 whitespace-nowrap"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={i % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
              {(row as unknown[]).map((cell, j) => (
                <td key={j} className="px-2 py-1 border-b border-gray-100 whitespace-nowrap">
                  {cell !== null && cell !== undefined ? String(cell) : '—'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      role: 'bot',
      text:
        'Xin chào! Tôi là trợ lý thống kê ngủ gật. ' +
        'Hỏi tôi về dữ liệu camera, học sinh, thời gian... bằng tiếng Việt. ' +
        'Hoặc chọn câu hỏi gợi ý ở trên.',
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll xuống cuối khi có tin nhắn mới
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendQuestion = async (question: string) => {
    const q = question.trim();
    if (!q || loading) return;

    setInput('');

    const userMsg: UserMessage = { id: `u-${Date.now()}`, role: 'user', text: q };
    // Placeholder loading message (sẽ được thay thế khi có kết quả)
    const loadingId = `b-${Date.now()}`;
    const loadingMsg: BotMessage = { id: loadingId, role: 'bot', text: '', loading: true };

    setMessages((prev) => [...prev, userMsg, loadingMsg]);
    setLoading(true);

    try {
      const res = await apiPost('/api/chatbot/query', { question: q });
      const data = await res.json();

      const botMsg: BotMessage = {
        id: loadingId,
        role: 'bot',
        text: data.summary_text || (data.success ? 'Không có dữ liệu.' : (data.error ?? 'Lỗi không xác định.')),
        columnNames: data.column_names,
        rows: data.rows,
        chartSuggestion: data.chart_suggestion,
        riskLevel: data.risk_level,
        loading: false,
        error: !data.success,
      };

      setMessages((prev) => prev.map((m) => (m.id === loadingId ? botMsg : m)));
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadingId
            ? {
                ...(m as BotMessage),
                loading: false,
                text: 'Lỗi kết nối tới backend. Hãy kiểm tra Python server.',
                error: true,
              }
            : m
        )
      );
    } finally {
      setLoading(false);
      setTimeout(() => inputRef.current?.focus(), 80);
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendQuestion(input);
    }
  };

  return (
    <div className="flex flex-col h-full bg-background overflow-hidden">

      {/* ── Quick suggestion buttons ── */}
      <div className="flex items-center gap-2 flex-wrap px-4 py-2 border-b bg-muted/20 flex-shrink-0">
        <span className="text-xs text-muted-foreground font-medium">Hỏi nhanh:</span>
        {QUICK_QUESTIONS.map((item) => (
          <button
            key={item.label}
            onClick={() => sendQuestion(item.q)}
            disabled={loading}
            className="text-xs px-2.5 py-1 rounded-full border border-border bg-background hover:bg-accent disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {item.label}
          </button>
        ))}
      </div>

      {/* ── Message list ── */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.map((msg) => {
          const isUser = msg.role === 'user';
          const bot = msg as BotMessage;

          return (
            <div key={msg.id} className={`flex gap-2.5 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
              {/* Avatar */}
              <div
                className={`flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-white
                  ${isUser ? 'bg-blue-500' : 'bg-slate-600'}`}
              >
                {isUser ? <User size={13} /> : <Bot size={13} />}
              </div>

              {/* Bubble + table */}
              <div className={`max-w-[80%] flex flex-col gap-1 ${isUser ? 'items-end' : 'items-start'}`}>
                <div
                  className={`rounded-2xl px-3.5 py-2 text-sm leading-relaxed
                    ${isUser
                      ? 'bg-blue-500 text-white rounded-tr-sm'
                      : bot.error
                      ? 'bg-red-50 border border-red-200 text-red-700 rounded-tl-sm'
                      : 'bg-card border border-border text-foreground rounded-tl-sm'
                    }`}
                >
                  {bot.loading ? (
                    <span className="flex items-center gap-2 text-muted-foreground text-xs">
                      <Loader2 size={13} className="animate-spin" />
                      Đang truy vấn dữ liệu...
                    </span>
                  ) : (
                    <span>
                      {bot.error && !isUser && (
                        <AlertCircle size={13} className="inline mr-1 align-middle" />
                      )}
                      {msg.text}
                      {!isUser && bot.riskLevel && bot.riskLevel !== 'none' && (
                        <RiskBadge level={bot.riskLevel} />
                      )}
                    </span>
                  )}
                </div>

                {/* Bảng kết quả */}
                {!bot.loading && !isUser && bot.columnNames?.length && bot.rows?.length ? (
                  <ResultTable columns={bot.columnNames} rows={bot.rows} />
                ) : null}
              </div>
            </div>
          );
        })}
        <div ref={bottomRef} />
      </div>

      {/* ── Input bar ── */}
      <div className="border-t px-4 py-3 flex gap-2 flex-shrink-0 bg-background">
        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Nhập câu hỏi bằng tiếng Việt... (Enter để gửi)"
          disabled={loading}
          className="flex-1 text-sm px-3 py-2 rounded-lg border border-border bg-background focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:opacity-50 placeholder:text-muted-foreground"
        />
        <button
          onClick={() => sendQuestion(input)}
          disabled={loading || !input.trim()}
          className="flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-blue-500 text-white text-sm font-medium hover:bg-blue-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? <Loader2 size={15} className="animate-spin" /> : <Send size={15} />}
          Gửi
        </button>
      </div>
    </div>
  );
}
