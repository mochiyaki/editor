import { useEffect, useState } from "react";
import { EditorApp } from "./editor/EditorApp";

type Theme = "dark" | "light";

const THEME_STORAGE_KEY = "gguf-editor-theme";
const LEGACY_THEME_STORAGE_KEY = "gguf-desktop-theme";

export default function App() {
  const [theme, setTheme] = useState<Theme>(
    () =>
      ((localStorage.getItem(THEME_STORAGE_KEY) ||
        localStorage.getItem(LEGACY_THEME_STORAGE_KEY)) as Theme) || "dark"
  );

  useEffect(() => {
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  return (
    <div className="flex h-screen overflow-hidden">
      <EditorApp
        theme={theme}
        visible
        onToggleTheme={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
      />
    </div>
  );
}
