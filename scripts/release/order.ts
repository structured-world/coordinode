/**
 * Locale-independent string order for reports, evidence and SBOMs: the same
 * input gives the same bytes on every runner, which `localeCompare` does not
 * promise.
 */
export const byCodeUnit = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
