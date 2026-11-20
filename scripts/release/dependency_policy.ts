// Dependency checks shared by preparation and registry-only verification.
import { byCodeUnit } from "./order.ts";

interface NamedPackage {
  name: string;
}

/** Reject the known native stack helpers from the delivered graph. */
export function checkNativeStackHelpers(packages: NamedPackage[]): void {
  const native = [...new Set(packages.map((p) => p.name).filter((n) => n === "stacker" || n === "psm"))].sort(byCodeUnit);
  if (native.length > 0) {
    throw new Error(`registry closure contains a native stack helper: ${JSON.stringify(native)}`);
  }
}
