/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import React, { createContext, useContext, useState, ReactNode } from "react";

export interface RetrievedDoc {
  file_path: string;
  score: number| null;
  text: string;
}

export type PromptLogEntry = {
  prompt: string;
  response: string;
  source: string;
  timestamp: number;
  model?: string;      // Model name/ID
  timeTaken?: number;  // in seconds or ms
    retrieval?: {
      rag_query: string;
      docs: RetrievedDoc[];
       // optional: the fully formatted context text block
    };
};

type PromptLogContextType = {
  promptLog: PromptLogEntry[];
  addPromptLog: (entry: PromptLogEntry) => void;
};

const PromptLogContext = createContext<PromptLogContextType | undefined>(
  undefined
);

export const PromptLogProvider: React.FC<{ children: ReactNode }> = ({
  children,
}) => {
  const [promptLog, setPromptLog] = useState<PromptLogEntry[]>([]);

  const addPromptLog = (entry: PromptLogEntry) => {
    setPromptLog((log) => [...log, entry]);
  //       console.log(
  //               `entry  : ${entry}`
  //             );
  //   console.log(
  //               `retrieval docs : ${entry.retrieval?.docs?.length}`
  //             );
  };

  return (
    <PromptLogContext.Provider value={{ promptLog, addPromptLog, setPromptLog }}>
      {children}
    </PromptLogContext.Provider>
  );
};

export const usePromptLog = () => {
  const context = useContext(PromptLogContext);
  if (!context)
    throw new Error("usePromptLog must be used within a PromptLogProvider");
  return context;
};
