import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist", "playwright-report", "test-results", "src/api/generated"] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // This application is not compiled with React Compiler; TanStack Table and
      // React Hook Form intentionally return mutable functions as part of their APIs.
      "react-hooks/incompatible-library": "off",
      "react-refresh/only-export-components": ["warn", { "allowConstantExport": true }],
      "@typescript-eslint/no-explicit-any": "error"
    },
  },
  {
    files: ["tests/**/*.ts", "playwright.config.ts", "vite.config.ts"],
    languageOptions: { globals: { ...globals.node, ...globals.browser } },
  },
);
