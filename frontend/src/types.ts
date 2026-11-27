export type Status = "pending" | "in_progress" | "completed";
export type Priority = "low" | "medium" | "high";
export type Role = "user" | "admin";
export interface Task {
  id: number; title: string; description: string | null;
  status: Status; priority: Priority; due_date: string | null; created_at: string;
}
export type TaskInput = Pick<Task, "title" | "description" | "priority" | "due_date"> & { status?: Status };

/** Admin view: one registered user with their task counts per status. */
export interface AdminUser {
  id: number; name: string; email: string; role: Role; created_at: string;
  pending: number; in_progress: number; completed: number; total: number;
}
