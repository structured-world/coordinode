<script setup lang="ts">
import { ref, computed, nextTick, onMounted, onUnmounted } from "vue";
import { useRoute } from "vitepress";

const MIN_DESCRIPTION_LENGTH = 10;

const CATEGORIES = [
  "Documentation is wrong/outdated",
  "Example query does not work",
  "Missing information",
  "Installation/setup issue",
  "Other",
];

const route = useRoute();

type WidgetState = "collapsed" | "expanded" | "submitting" | "success" | "error";

const state = ref<WidgetState>("collapsed");
const description = ref("");
const expected = ref("");
const category = ref("");
const honeypot = ref("");
const errorMessage = ref("");
const panelRef = ref<HTMLElement | null>(null);
const triggerRef = ref<HTMLElement | null>(null);

const categories = CATEGORIES;
const isValid = computed(() => description.value.trim().length >= MIN_DESCRIPTION_LENGTH);
const currentPage = computed(() => route.path);

function toggle() {
  if (state.value === "collapsed") {
    state.value = "expanded";
    focusPanel();
  } else if (state.value !== "collapsed") {
    state.value = "collapsed";
    resetForm();
    nextTick(() => triggerRef.value?.focus());
  }
}

function resetForm() {
  description.value = "";
  expected.value = "";
  category.value = "";
  honeypot.value = "";
  errorMessage.value = "";
}

async function submit() {
  if (!isValid.value) return;
  state.value = "submitting";
  errorMessage.value = "";

  try {
    const response = await fetch("/api/report-bug", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        page: currentPage.value,
        description: description.value.trim(),
        expected: expected.value.trim() || undefined,
        category: category.value || undefined,
        honeypot: honeypot.value,
      }),
    });

    if (response.ok) {
      state.value = "success";
      resetForm();
      setTimeout(() => { state.value = "collapsed"; }, 3000);
    } else {
      const data = await response.json().catch(() => ({}));
      errorMessage.value = (data as { error?: string }).error || "Something went wrong";
      state.value = "error";
    }
  } catch {
    errorMessage.value = "Network error. Please try again.";
    state.value = "error";
  }
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === "Escape" && state.value !== "collapsed") {
    state.value = "collapsed";
    resetForm();
    nextTick(() => triggerRef.value?.focus());
    return;
  }
  if (e.key === "Tab" && state.value !== "collapsed" && panelRef.value) {
    const focusable = panelRef.value.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (e.shiftKey && document.activeElement === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  }
}

function focusPanel() {
  nextTick(() => {
    const firstInput = panelRef.value?.querySelector<HTMLElement>("textarea, button, input");
    firstInput?.focus();
  });
}

onMounted(() => document.addEventListener("keydown", handleKeydown));
onUnmounted(() => document.removeEventListener("keydown", handleKeydown));
</script>

<template>
  <div class="bug-report-widget" :class="{ expanded: state !== 'collapsed' }">
    <button
      v-if="state === 'collapsed'"
      ref="triggerRef"
      class="bug-tab"
      @click="toggle"
      aria-label="Report a bug"
      title="Report a bug"
    >
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
           fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
           stroke-linejoin="round" aria-hidden="true">
        <path d="M8 2l1.88 1.88"/><path d="M14.12 3.88L16 2"/>
        <path d="M9 7.13v-1a3.003 3.003 0 1 1 6 0v1"/>
        <path d="M12 20c-3.3 0-6-2.7-6-6v-3a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v3c0 3.3-2.7 6-6 6"/>
        <path d="M12 20v-9"/><path d="M6.53 9C4.6 8.8 3 7.1 3 5"/>
        <path d="M6 13H2"/><path d="M3 21c0-2.1 1.7-3.9 3.8-4"/>
        <path d="M20.97 5c0 2.1-1.6 3.8-3.5 4"/><path d="M22 13h-4"/>
        <path d="M17.2 17c2.1.1 3.8 1.9 3.8 4"/>
      </svg>
      <span class="bug-tab-text">Report Bug</span>
    </button>

    <div v-if="state !== 'collapsed'" ref="panelRef" class="bug-panel"
         role="dialog" aria-modal="true" aria-label="Bug report form">
      <div class="bug-panel-header">
        <span class="bug-panel-title">Found an issue?</span>
        <button class="bug-panel-close" @click="toggle" aria-label="Close">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
               fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
               stroke-linejoin="round" aria-hidden="true">
            <path d="M18 6L6 18"/><path d="M6 6l12 12"/>
          </svg>
        </button>
      </div>

      <div v-if="state === 'success'" class="bug-success">
        <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24"
             fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
             stroke-linejoin="round" class="checkmark" aria-hidden="true">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
          <polyline points="22 4 12 14.01 9 11.01"/>
        </svg>
        <p>Thanks! We'll look into it.</p>
      </div>

      <form v-else @submit.prevent="submit" class="bug-form">
        <input v-model="honeypot" type="text" name="website" autocomplete="off"
               tabindex="-1" class="bug-honeypot" aria-hidden="true"/>

        <div class="bug-field">
          <label for="bug-description">What happened? <span class="required">*</span></label>
          <textarea id="bug-description" v-model="description"
                    placeholder="Describe what was wrong or confusing..."
                    rows="3" required :disabled="state === 'submitting'"/>
          <span v-if="description.length > 0 && description.trim().length < MIN_DESCRIPTION_LENGTH" class="field-hint">
            At least {{ MIN_DESCRIPTION_LENGTH }} characters required
          </span>
        </div>

        <div class="bug-field">
          <label for="bug-expected">What did you expect? <span class="optional">(optional)</span></label>
          <textarea id="bug-expected" v-model="expected"
                    placeholder="What should have happened instead..."
                    rows="2" :disabled="state === 'submitting'"/>
        </div>

        <div class="bug-field">
          <label for="bug-category">Category <span class="optional">(optional)</span></label>
          <select id="bug-category" v-model="category" :disabled="state === 'submitting'">
            <option value="">Select...</option>
            <option v-for="cat in categories" :key="cat" :value="cat">{{ cat }}</option>
          </select>
        </div>

        <div v-if="state === 'error' && errorMessage" class="bug-error">{{ errorMessage }}</div>

        <div class="bug-actions">
          <button type="submit" class="bug-submit" :disabled="!isValid || state === 'submitting'">
            <span v-if="state === 'submitting'" class="spinner" aria-hidden="true"/>
            {{ state === "submitting" ? "Sending..." : "Send Report" }}
          </button>
        </div>

        <div class="bug-footer">
          <a href="https://github.com/structured-world/coordinode/issues/new?template=bug.yml"
             target="_blank" rel="noopener">
            Or report on GitHub
          </a>
        </div>
      </form>
    </div>
  </div>
</template>

<style scoped>
.bug-report-widget { position: fixed; right: 24px; bottom: 24px; z-index: 100; }

.bug-tab {
  display: flex; align-items: center; gap: 6px; padding: 12px 16px;
  background: var(--vp-c-brand-1); color: #fff; border: none; border-radius: 24px;
  cursor: pointer; font-size: 14px; font-weight: 600;
  transition: transform 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
  box-shadow: 0 4px 16px rgba(0,0,0,0.2); animation: pulse 3s ease-in-out 2s 3;
}
.bug-tab:hover { transform: translateY(-2px); background: var(--vp-c-brand-2); box-shadow: 0 6px 20px rgba(0,0,0,0.25); }
@keyframes pulse { 0%, 100% { box-shadow: 0 4px 16px rgba(0,0,0,0.2); } 50% { box-shadow: 0 4px 16px rgba(0,0,0,0.2), 0 0 0 6px rgba(232,93,44,0.2); } }

.bug-panel {
  position: fixed; right: 24px; bottom: 80px; width: 340px; max-height: 70vh;
  overflow-y: auto; background: var(--vp-c-bg); border: 1px solid var(--vp-c-border);
  border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.15); padding: 16px;
  animation: slideIn 0.2s ease;
}
@keyframes slideIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
@media (prefers-reduced-motion: reduce) { .bug-panel { animation: none; } .bug-tab { transition: none; } }

.bug-panel-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
.bug-panel-title { font-size: 16px; font-weight: 600; color: var(--vp-c-text-1); }
.bug-panel-close { background: none; border: none; cursor: pointer; color: var(--vp-c-text-3); padding: 4px; border-radius: 4px; display: flex; align-items: center; }
.bug-panel-close:hover { color: var(--vp-c-text-1); background: var(--vp-c-bg-soft); }

.bug-form { display: flex; flex-direction: column; gap: 12px; }
.bug-field { display: flex; flex-direction: column; gap: 4px; }
.bug-field label { font-size: 13px; font-weight: 500; color: var(--vp-c-text-2); }
.required { color: var(--vp-c-danger-1); }
.optional { font-weight: 400; color: var(--vp-c-text-3); font-size: 12px; }
.bug-field textarea, .bug-field select { font-family: inherit; font-size: 14px; padding: 8px 10px; border: 1px solid var(--vp-c-border); border-radius: 6px; background: var(--vp-c-bg-soft); color: var(--vp-c-text-1); resize: vertical; transition: border-color 0.2s; }
.bug-field textarea:focus, .bug-field select:focus { outline: none; border-color: var(--vp-c-brand-1); }
.bug-field textarea:disabled, .bug-field select:disabled { opacity: 0.6; cursor: not-allowed; }
.field-hint { font-size: 12px; color: var(--vp-c-danger-1); }
.bug-honeypot { position: absolute; left: -9999px; width: 1px; height: 1px; overflow: hidden; opacity: 0; }

.bug-actions { margin-top: 4px; }
.bug-submit { width: 100%; padding: 10px 16px; background: var(--vp-c-brand-1); color: #fff; border: none; border-radius: 8px; font-size: 14px; font-weight: 500; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; transition: background 0.2s; }
.bug-submit:hover:not(:disabled) { background: var(--vp-c-brand-2); }
.bug-submit:disabled { opacity: 0.5; cursor: not-allowed; }
.spinner { width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.3); border-top-color: #fff; border-radius: 50%; animation: spin 0.6s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.bug-error { padding: 8px 10px; background: var(--vp-c-danger-soft); color: var(--vp-c-danger-1); border-radius: 6px; font-size: 13px; }
.bug-success { display: flex; flex-direction: column; align-items: center; gap: 12px; padding: 24px 16px; text-align: center; }
.bug-success .checkmark { color: var(--vp-c-brand-1); animation: scaleIn 0.3s ease; }
@keyframes scaleIn { from { transform: scale(0.5); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.bug-success p { font-size: 15px; color: var(--vp-c-text-1); margin: 0; }
.bug-footer { text-align: center; padding-top: 4px; }
.bug-footer a { font-size: 12px; color: var(--vp-c-text-3); text-decoration: none; }
.bug-footer a:hover { color: var(--vp-c-brand-1); text-decoration: underline; }

@media (max-width: 768px) {
  .bug-report-widget { right: 16px; bottom: 16px; }
  .bug-tab { border-radius: 50%; width: 48px; height: 48px; padding: 0; justify-content: center; }
  .bug-tab svg { width: 20px; height: 20px; }
  .bug-tab-text { display: none; }
  .bug-panel { right: 8px; left: 8px; bottom: 8px; width: auto; }
}
</style>
