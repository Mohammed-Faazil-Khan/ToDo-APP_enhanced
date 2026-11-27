import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { Legend, StatCard, StatusBar, TaskRow } from "../components";
import type { Task } from "../types";

export default function Dashboard() {
  const [tasks, setTasks] = useState<Task[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => { api.listTasks().then(setTasks).catch((e) => setError(e.message)); }, []);

  if (error) return <div className="error">{error}</div>;
  if (!tasks) return <p className="muted">Loading your tasks…</p>;

  const count = (s: Task["status"]) => tasks.filter((t) => t.status === s).length;
  const pending = count("pending"), progress = count("in_progress"), done = count("completed");
  const total = tasks.length;
  const pct = total ? Math.round((done / total) * 100) : 0;
  const recent = [...tasks].sort((a, b) => b.id - a.id).slice(0, 5); // newest first (ids grow with creation)

  return (
    <>
      <div className="stats">
        <StatCard label="Total Tasks" value={total} hint="Tasks in workspace" icon="☷" />
        <StatCard label="Pending" value={pending} hint="Waiting to be started" icon="◷" />
        <StatCard label="In Progress" value={progress} hint="Currently being worked on" icon="↗" />
        <StatCard label="Completed" value={done} hint="Successfully finished" icon="✓" />
      </div>

      {total > 0 && (
        <section className="panel panel-body" aria-label="Progress">
          <div className="progress-head"><span className="big">{pct}%</span><span className="muted">{done} of {total} tasks completed</span></div>
          <StatusBar pending={pending} in_progress={progress} completed={done} />
          <Legend pending={pending} in_progress={progress} completed={done} />
        </section>
      )}

      <section className="panel">
        <div className="panel-header">
          <div><h2>Recent Tasks</h2><p>Your latest tasks at a glance.</p></div>
          <Link className="btn btn-secondary" to="/tasks">View All →</Link>
        </div>
        {recent.length === 0 ? (
          <div className="empty"><h3>No tasks yet</h3><Link to="/tasks?new=1">Add your first task</Link></div>
        ) : (
          <div className="task-list">{recent.map((t) => <TaskRow key={t.id} task={t} />)}</div>
        )}
      </section>
    </>
  );
}
