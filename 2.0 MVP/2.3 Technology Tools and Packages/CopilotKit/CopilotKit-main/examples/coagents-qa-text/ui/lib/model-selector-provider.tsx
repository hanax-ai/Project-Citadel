"use client";

import React from "react";
import { createContext, useContext, useState, ReactNode } from "react";

type ModelSelectorContextType = {
  model: string;
  setModel: (model: string) => void;
  hidden: boolean;
  lgcDeploymentUrl?: string | null;
  setHidden: (hidden: boolean) => void;
  agent: string;
};

const ModelSelectorContext = createContext<
  ModelSelectorContextType | undefined
>(undefined);

export const ModelSelectorProvider = ({
  children,
}: {
  children: ReactNode;
}) => {
  const model =
    globalThis.window === undefined
      ? "openai"
      : new URL(window.location.href).searchParams.get("coAgentsModel") ??
        "openai";
  const [hidden, setHidden] = useState<boolean>(false);

  const setModel = (model: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set("coAgentsModel", model);
    window.location.href = url.toString();
  };

  const lgcDeploymentUrl =
    globalThis.window === undefined
      ? null
      : new URL(window.location.href).searchParams.get("lgcDeploymentUrl");

  const agent = model === "crewai" ? "greeting_agent_crewai" : "greeting_agent";

  return (
    <ModelSelectorContext.Provider
      value={{
        model,
        hidden,
        lgcDeploymentUrl,
        setModel,
        setHidden,
        agent,
      }}
    >
      {children}
    </ModelSelectorContext.Provider>
  );
};

export const useModelSelectorContext = () => {
  const context = useContext(ModelSelectorContext);
  if (context === undefined) {
    throw new Error(
      "useModelSelectorContext must be used within a ModelSelectorProvider"
    );
  }
  return context;
};
