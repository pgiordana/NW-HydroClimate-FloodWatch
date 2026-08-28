const OWNER = "pgiordana";
const REPO = "NW-HydroClimate-FloodWatch";
const WORKFLOW = "nw-floodwatch-daily.yml";
const REF = "main";
const PUBLIC_STATE_URL = "https://nw-floodwatch.pages.dev/data/latest.json";
const ROME_TZ = "Europe/Rome";
const TARGET_TIMES = new Set(["10:17", "11:00"]);

function romeParts(date = new Date()) {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: ROME_TZ,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
  }).formatToParts(date);

  const get = (type) => parts.find((p) => p.type === type)?.value;
  return {
    date: `${get("year")}-${get("month")}-${get("day")}`,
    time: `${get("hour")}:${get("minute")}`,
  };
}

async function githubRequest(env, path, init = {}) {
  if (!env.GITHUB_ACTIONS_TOKEN) {
    throw new Error("Missing Cloudflare secret GITHUB_ACTIONS_TOKEN");
  }

  const headers = new Headers(init.headers || {});
  headers.set("Accept", "application/vnd.github+json");
  headers.set("Authorization", `Bearer ${env.GITHUB_ACTIONS_TOKEN}`);
  headers.set("X-GitHub-Api-Version", "2022-11-28");
  headers.set("User-Agent", "NW-FloodWatch-Cloudflare-Scheduler");

  return fetch(`https://api.github.com${path}`, { ...init, headers });
}

async function siteAlreadyCurrent(today) {
  try {
    const response = await fetch(`${PUBLIC_STATE_URL}?scheduler=${Date.now()}`, {
      headers: {
        "Cache-Control": "no-cache",
        "User-Agent": "NW-FloodWatch-Cloudflare-Scheduler",
      },
    });
    if (!response.ok) return false;
    const data = await response.json();
    return data?.issue_date === today;
  } catch {
    // Fail open: inability to read the public site must not suppress production.
    return false;
  }
}

async function workflowAlreadyRunning(env, today) {
  const path = `/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW}/runs?branch=${REF}&per_page=20`;
  const response = await githubRequest(env, path);
  if (!response.ok) {
    throw new Error(`GitHub runs check failed: ${response.status} ${await response.text()}`);
  }

  const payload = await response.json();
  return (payload.workflow_runs || []).some((run) => {
    const created = romeParts(new Date(run.created_at)).date;
    return created === today && (run.status === "queued" || run.status === "in_progress");
  });
}

async function dispatchWorkflow(env) {
  const path = `/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW}/dispatches`;
  const response = await githubRequest(env, path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ref: REF }),
  });

  if (response.status !== 204) {
    throw new Error(`GitHub workflow dispatch failed: ${response.status} ${await response.text()}`);
  }
}

async function runScheduler(env, now = new Date()) {
  const { date: today, time } = romeParts(now);

  // Four UTC cron triggers are configured to cover CET and CEST.
  // Only the trigger matching the intended Europe/Rome local time is allowed to dispatch.
  if (!TARGET_TIMES.has(time)) {
    return { action: "noop", reason: `local_time_${time}_not_target`, today, time };
  }

  if (await siteAlreadyCurrent(today)) {
    return { action: "noop", reason: "site_already_current", today, time };
  }

  if (await workflowAlreadyRunning(env, today)) {
    return { action: "noop", reason: "workflow_already_running", today, time };
  }

  await dispatchWorkflow(env);
  return { action: "dispatched", today, time, workflow: WORKFLOW, ref: REF };
}

export default {
  async scheduled(controller, env, ctx) {
    ctx.waitUntil(
      runScheduler(env, new Date(controller.scheduledTime)).then((result) => {
        console.log(JSON.stringify(result));
      })
    );
  },

  async fetch(request, env) {
    const url = new URL(request.url);
    const { date: today, time } = romeParts(new Date());

    // Safe, read-only diagnostic. It validates that the Worker can read GitHub Actions
    // with the configured secret, but it never dispatches a workflow.
    if (url.searchParams.get("check") === "1") {
      const siteCurrent = await siteAlreadyCurrent(today);
      try {
        const workflowRunning = await workflowAlreadyRunning(env, today);
        return Response.json({
          service: "NW FloodWatch Cloudflare Scheduler",
          status: "ok",
          github_auth: "ok",
          site_current: siteCurrent,
          workflow_running: workflowRunning,
          today,
          local_time: time,
          timezone: ROME_TZ,
          target_local_times: [...TARGET_TIMES],
        });
      } catch (error) {
        return Response.json(
          {
            service: "NW FloodWatch Cloudflare Scheduler",
            status: "error",
            github_auth: "failed",
            message: String(error?.message || error),
            today,
            local_time: time,
            timezone: ROME_TZ,
          },
          { status: 500 }
        );
      }
    }

    return Response.json({
      service: "NW FloodWatch Cloudflare Scheduler",
      mode: "scheduled-only",
      timezone: ROME_TZ,
      target_local_times: [...TARGET_TIMES],
      workflow: `${OWNER}/${REPO} :: ${WORKFLOW}@${REF}`,
      status: "ok",
      health_check: "append ?check=1 for a read-only GitHub authorization test",
    });
  },
};
