import { useCallback, useEffect, useState, type FormEvent } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "../api";
import { STATUS_LABEL, TaskRow } from "../components";
import type { Priority, Status, Task } from "../types";

const PRI = { high: 0, medium: 1, low: 2 };
/** The backend may ignore status / q / sort, so apply them here too (harmless if it already did). */
function refine(list: Task[], status: string, q: string, sort: string): Task[] {
  const needle = q.trim().toLowerCase();
  const out = list.filter((t) => (!status || t.status === status) && (!needle || t.title.toLowerCase().includes(needle)));
  if (sort === "due_date") out.sort((a, b) => (a.due_date ?? "9999").localeCompare(b.due_date ?? "9999"));
  else if (sort === "priority") out.sort((a, b) => PRI[a.priority] - PRI[b.priority]);
  else out.sort((a, b) => b.id - a.id);
  return out;
}
const blank = { title: "", description: "", priority: "medium" as Priority, due_date: "", status: "pending" as Status };

export default function Tasks() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [status, setStatus] = useState(""); const [priority, setPriority] = useState(""); const [q, setQ] = useState(""); const [sort, setSort] = useState("created_at");
  const [loading, setLoading] = useState(true); const [error, setError] = useState("");
  const [form, setForm] = useState(blank); const [editId, setEditId] = useState<number | null>(null); const [showForm, setShowForm] = useState(false);
  const [params, setParams] = useSearchParams();
  // AI task breakdown
  const [aiGoal, setAiGoal] = useState(""); const [breakdownBusy, setBreakdownBusy] = useState(false);
  const [aiSuggestions, setAiSuggestions] = useState<{ title: string; description: string; priority: Priority }[]>([]);

  const load = useCallback(async () => {
    try { setError(""); setTasks(refine(await api.listTasks({ status, q, sort }), status, q, sort)); }
    catch (e) { setError(e instanceof Error ? e.message : "Could not load tasks."); }
    finally { setLoading(false); }
  }, [status, q, sort]);

  useEffect(() => { const t = setTimeout(load, 250); return () => clearTimeout(t); }, [load]); // debounce search

  const openNew = useCallback(() => { setForm(blank); setEditId(null); setShowForm(true); }, []);
  // "Add a Task" / "New Task" buttons elsewhere link here with ?new=1
  useEffect(() => { if (params.get("new")) { openNew(); setParams({}, { replace: true }); } }, [params, openNew, setParams]);

  const openEdit = (t: Task) => {
    setForm({ title: t.title, description: t.description ?? "", priority: t.priority, due_date: t.due_date ?? "", status: t.status });
    setEditId(t.id); setShowForm(true);
  };

  const save = async (e: FormEvent) => {
    e.preventDefault();
    if (!form.title.trim()) return setError("Give the task a title.");
    const body = { title: form.title.trim(), description: form.description.trim() || null, priority: form.priority, due_date: form.due_date || null, status: form.status };
    try {
      editId ? await api.updateTask(editId, body) : await api.createTask(body);
      setShowForm(false); await load();
    } catch (err) { setError(err instanceof Error ? err.message : "Could not save the task."); }
  };

  const askAi = async (e: FormEvent) => {
    e.preventDefault();
    if (aiGoal.trim().length < 3) return setError("Describe your goal in a few words first.");
    setBreakdownBusy(true); setError(""); setAiSuggestions([]);
    try { setAiSuggestions((await api.aiBreakdown(aiGoal.trim())).subtasks); }
    catch (err) { setError(err instanceof Error ? err.message : "AI request failed."); }
    finally { setBreakdownBusy(false); }
  };
  const addSuggestions = async () => {
    try {
      for (const s of aiSuggestions) await api.createTask({ title: s.title, description: s.description || null, priority: s.priority, due_date: null, status: "pending" });
      setAiSuggestions([]); setAiGoal(""); await load();
    } catch (err) { setError(err instanceof Error ? err.message : "Could not add the tasks."); }
  };

  const setTaskStatus = async (t: Task, s: Status) => {
    setTasks((prev) => prev.map((x) => (x.id === t.id ? { ...x, status: s } : x))); // optimistic
    try { await api.updateTask(t.id, { status: s }); } catch (e) { setError(e instanceof Error ? e.message : "Update failed."); load(); }
  };
  const remove = async (t: Task) => {
    if (!confirm(`Delete "${t.title}"?`)) return;
    try { await api.deleteTask(t.id); setTasks((p) => p.filter((x) => x.id !== t.id)); }
    catch (e) { setError(e instanceof Error ? e.message : "Delete failed."); }
  };

  const shown = priority ? tasks.filter((t) => t.priority === priority) : tasks; // priority filter is client-side

  return (
    <>
      {error && <div className="error" role="alert">{error}</div>}
      <section className="panel">
        <div className="panel-header">
          <div><h2>Task List</h2><p>Search, filter and manage your tasks.</p></div>
          <span className="muted">{shown.length} {shown.length === 1 ? "task" : "tasks"}</span>
        </div>
        <form className="toolbar" onSubmit={askAi}>
          <input type="text" placeholder="✨ Describe a goal, e.g. &quot;prepare for my viva&quot;" value={aiGoal} maxLength={200}
            onChange={(e) => setAiGoal(e.target.value)} aria-label="Goal for AI breakdown" />
          <button className="btn btn-primary" disabled={breakdownBusy}>{breakdownBusy ? "Thinking…" : "Break down with AI"}</button>
        </form>
        {aiSuggestions.length > 0 && (
          <div className="panel" style={{ margin: "12px 0" }}>
            <h3>AI suggestions</h3>
            <ul>{aiSuggestions.map((s, i) => <li key={i}><strong>{s.title}</strong> <span className="muted">({s.priority})</span>{s.description && <> — {s.description}</>}</li>)}</ul>
            <button className="btn btn-primary" onClick={addSuggestions}>Add all {aiSuggestions.length} as tasks</button>{" "}
            <button className="btn btn-secondary" onClick={() => setAiSuggestions([])}>Dismiss</button>
          </div>
        )}
        <div className="toolbar">
          <input type="search" placeholder="⌕ Search tasks by title..." value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search tasks" />
          <select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Filter by status">
            <option value="">All Statuses</option>
            {Object.entries(STATUS_LABEL).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
          <select value={priority} onChange={(e) => setPriority(e.target.value)} aria-label="Filter by priority">
            <option value="">All Priorities</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option>
          </select>
          <select value={sort} onChange={(e) => setSort(e.target.value)} aria-label="Sort tasks">
            <option value="created_at">Newest First</option><option value="due_date">Due Date</option><option value="priority">Priority</option>
          </select>
        </div>

        {loading ? <p className="empty">Loading tasks…</p> : shown.length === 0 ? (
          <div className="empty"><h3>{q || status || priority ? "No tasks match these filters" : "No tasks yet"}</h3>{!(q || status || priority) && "Add one to get started."}</div>
        ) : (
          <div className="task-list">
            {shown.map((t) => (
              <TaskRow key={t.id} task={t} onToggle={() => setTaskStatus(t, t.status === "completed" ? "pending" : "completed")}
                actions={<>
                  {t.status === "pending" && <button className="btn btn-secondary btn-sm" onClick={() => setTaskStatus(t, "in_progress")}>Start</button>}
                  <button className="btn btn-secondary btn-sm" onClick={() => openEdit(t)}>Edit</button>
                  <button className="btn btn-danger btn-sm" onClick={() => remove(t)}>Delete</button>
                </>} />
            ))}
          </div>
        )}
      </section>

      {showForm && (
        <div className="modal-backdrop" onClick={(e) => { if (e.target === e.currentTarget) setShowForm(false); }}>
          <div className="modal" role="dialog" aria-modal="true" aria-labelledby="mt">
            <div className="modal-header">
              <div><h2 id="mt">{editId ? "Edit Task" : "Create a Task"}</h2><p className="subtitle">Add details and set a priority.</p></div>
              <button type="button" className="icon-button" onClick={() => setShowForm(false)} aria-label="Close">✕</button>
            </div>
            <form onSubmit={save}>
              <div className="modal-body"><div className="modal-grid">
                <div className="field wide"><label htmlFor="tt">Task Title *</label><input id="tt" autoFocus required maxLength={120} placeholder="e.g. Complete project report" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} /></div>
                <div className="field wide"><label htmlFor="td">Description</label><textarea id="td" maxLength={500} placeholder="Add a few details..." value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} /></div>
                <div className="field"><label htmlFor="tp">Priority</label><select id="tp" value={form.priority} onChange={(e) => setForm({ ...form, priority: e.target.value as Priority })}><option value="medium">Medium</option><option value="high">High</option><option value="low">Low</option></select></div>
                <div className="field"><label htmlFor="ts">Status</label><select id="ts" value={form.status} onChange={(e) => setForm({ ...form, status: e.target.value as Status })}>{Object.entries(STATUS_LABEL).map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></div>
                <div className="field"><label htmlFor="tu">Due Date</label><input id="tu" type="date" value={form.due_date} onChange={(e) => setForm({ ...form, due_date: e.target.value })} /></div>
              </div></div>
              <div className="modal-footer">
                <button type="button" className="btn btn-secondary" onClick={() => setShowForm(false)}>Cancel</button>
                <button className="btn btn-primary">Save Task</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
