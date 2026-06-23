import { useEffect, useState, useRef } from "react";
import { Activity, Database, Clock, Zap } from "lucide-react";
import GlassCard from "../components/ui/GlassCard.jsx";
import SectionHeader from "../components/ui/SectionHeader.jsx";
import Badge from "../components/ui/Badge.jsx";
import { API_BASE } from "../lib/api.js";

export default function InteractionLogs() {
  const [logs, setLogs] = useState([]);
  const bottomRef = useRef(null);

  // Auto-scroll when new logs arrive
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  useEffect(() => {
    const sse = new EventSource(`${API_BASE}/logs/stream`);
    
    sse.addEventListener("message", (e) => {
      try {
        const data = JSON.parse(e.data);
        setLogs((prev) => {
          if (prev.find((l) => l.id === data.id)) return prev;
          return [...prev, data];
        });
      } catch (err) {
        console.error("Error parsing SSE log", err);
      }
    });

    return () => {
      sse.close();
    };
  }, []);

  return (
    <div className="space-y-4">
      <GlassCard className="p-5" accentBar="from-cyan to-violet">
        <div className="flex items-center justify-between mb-4">
          <SectionHeader title="Real-Time Interaction Logs" sub="Live stream from SQLite/Supabase" icon={Activity} accent="cyan" className="mb-0" />
          <Badge accent="emerald" pulse>Live SSE</Badge>
        </div>
        
        <div className="space-y-3 max-h-[70vh] overflow-y-auto pr-2">
          {logs.length === 0 && (
            <div className="text-sm text-ink-faint text-center py-10">
              No interactions yet. Ask the Tactical AI Commander a question.
            </div>
          )}
          {logs.map((log) => (
            <div key={log.id} className="rounded-inner border border-subtle bg-card/40 p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Database size={14} className="text-cyan" />
                  <span className="font-mono text-xs text-ink-faint">{new Date(log.timestamp).toLocaleString()}</span>
                </div>
                <div className="flex gap-2">
                  {log.intent && (
                    <Badge accent="violet">{log.intent}</Badge>
                  )}
                  {log.confidence_score > 0 && (
                    <Badge accent="emerald">Conf {log.confidence_score.toFixed(1)}%</Badge>
                  )}
                  {log.execution_time > 0 && (
                    <div className="flex items-center gap-1 text-[0.65rem] text-ink-faint">
                      <Clock size={12} /> {(log.execution_time * 1000).toFixed(0)}ms
                    </div>
                  )}
                </div>
              </div>
              
              <div className="mb-2">
                <span className="text-xs font-semibold text-ink-faint uppercase tracking-wider">User:</span>
                <p className="text-sm text-ink-primary mt-1">{log.user_query}</p>
              </div>
              
              <div className="mb-2">
                <span className="text-xs font-semibold text-ink-faint uppercase tracking-wider">Agent:</span>
                <p className="text-sm text-ink-body mt-1 line-clamp-3">{log.ai_response}</p>
              </div>

              {log.tools_used && log.tools_used.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2">
                  {log.tools_used.map((t) => (
                    <span key={t} className="flex items-center gap-1 rounded bg-base/50 px-2 py-0.5 text-[0.65rem] text-amber border border-amber/20">
                      <Zap size={10} /> {t}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </GlassCard>
    </div>
  );
}
