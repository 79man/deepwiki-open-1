/* eslint-disable @typescript-eslint/no-unused-vars */
"use client";

import Ask from "@/components/Ask";
import Markdown from "@/components/Markdown";
import ModelSelectionModal from "@/components/ModelSelectionModal";
import { ModelSelectionParams } from "@/types/modelSelection";
import ThemeToggle from "@/components/theme-toggle";
import WikiTreeView from "@/components/WikiTreeView";
import { useLanguage } from "@/contexts/LanguageContext";
import { RepoInfo } from "@/types/repoinfo";
import getRepoUrl from "@/utils/getRepoUrl";
import { extractUrlDomain, extractUrlPath } from "@/utils/urlDecoder";
import Link from "next/link";
import { useParams, useSearchParams } from "next/navigation";
import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  FaBitbucket,
  FaBookOpen,
  FaComments,
  FaDownload,
  FaExclamationTriangle,
  FaFileExport,
  FaFolder,
  FaGithub,
  FaGitlab,
  FaHome,
  FaSync,
  FaTimes,
} from "react-icons/fa";

import PromptEditorModal from "@/components/PromptEditorModal"; // Adjust import path if needed
import { usePromptLog } from "@/contexts/PromptLogContext";
import PromptLogFloatingPanel from "@/components/PromptLogFloatingPanel";
// import RetrievedDocsFloatingPanel from "@/components/RetrievedDocsFloatingPanel";
import {
  createChatWebSocket,
  closeWebSocket,
  ChatCompletionRequest,
} from "@/utils/websocketClient";

import IterationTabs from "@/components/IterationTabs";

// Define the WikiSection and WikiStructure types directly in this file
// since the imported types don't have the sections and rootSections properties
interface WikiSection {
  id: string;
  title: string;
  pages: string[];
  subsections?: string[];
}

import { WikiPage, WikiPageIteration } from "@/types/wiki/wikipage";

interface WikiStructure {
  id: string;
  title: string;
  description: string;
  pages: WikiPage[];
  sections: WikiSection[];
  rootSections: string[];
}

interface ReadmeAnalysis {
  project_purpose: string;
  key_features: string[];
  architecture_overview: string;
  main_technologies: string[];
  important_sections: string[];
}

type ReadmeContent = string | ReadmeAnalysis;

interface RepoBasicAnalysis {
  type: string;
  primaryLanguage: string;
  framework: string;
  architecturePattern: string;
  complexityScore: number;
}

interface RepoDeepAnalysis {
  domain: string;
  architecture_style: string;
  key_subsystems: Array<{
    name: string;
    purpose: string;
    entry_points: string[];
    dependencies: string[];
  }>;
  data_flow_patterns: string[];
  integration_points: string[];
  core_abstractions: string[];
  documentation_priorities: string[];
  domain_concepts: Record<string, string>;
}

interface Message {
  role: "user" | "assistant";
  content: string;
}

// Add CSS styles for wiki with Japanese aesthetic
const wikiStyles = `
  .prose code {
    @apply bg-[var(--background)]/70 px-1.5 py-0.5 rounded font-mono text-xs border border-[var(--border-color)];
  }

  .prose pre {
    @apply bg-[var(--background)]/80 text-[var(--foreground)] rounded-md p-4 overflow-x-auto border border-[var(--border-color)] shadow-sm;
  }

  .prose h1, .prose h2, .prose h3, .prose h4 {
    @apply font-serif text-[var(--foreground)];
  }

  .prose p {
    @apply text-[var(--foreground)] leading-relaxed;
  }

  .prose a {
    @apply text-[var(--accent-primary)] hover:text-[var(--highlight)] transition-colors no-underline border-b border-[var(--border-color)] hover:border-[var(--accent-primary)];
  }

  .prose blockquote {
    @apply border-l-4 border-[var(--accent-primary)]/30 bg-[var(--background)]/30 pl-4 py-1 italic;
  }

  .prose ul, .prose ol {
    @apply text-[var(--foreground)];
  }

  .prose table {
    @apply border-collapse border border-[var(--border-color)];
  }

  .prose th {
    @apply bg-[var(--background)]/70 text-[var(--foreground)] p-2 border border-[var(--border-color)];
  }

  .prose td {
    @apply p-2 border border-[var(--border-color)];
  }
  
  .prose .src-list {
    border-radius: var(--radius-md);
    border-style: var(--tw-border-style);
    border-width: 1px;
    border-color: var(--border-color);
    background-color: var(--background);
    padding-inline: calc(var(--spacing) * 3);
    padding-block: calc(var(--spacing) * 2);
    color: var(--foreground);
  }
  
  .prose details {
    border-radius: var(--radius-md);
    border-style: var(--tw-border-style);
    border-width: 1px;
    border-color: var(--border-color);
    background-color: var(--background);
    padding-inline: calc(var(--spacing) * 3);
    padding-block: calc(var(--spacing) * 2);
    color: var(--foreground);
    cursor: pointer;
  }
  .prose ul {
    margin-top: calc(var(--spacing) * 4);
  }
`;

// Helper function to generate cache key for localStorage
const getCacheKey = (
  owner: string,
  repo: string,
  repoType: string,
  language: string,
  isComprehensive: boolean = true
): string => {
  return `deepwiki_cache_${repoType}_${owner}_${repo}_${language}_${
    isComprehensive ? "comprehensive" : "concise"
  }`;
};

// Helper function to add tokens and other parameters to request body
const addTokensToRequestBody = (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  requestBody: ChatCompletionRequest,
  token: string,
  repoType: string,
  provider: string = "",
  model: string = "",
  isCustomModel: boolean = false,
  customModel: string = "",
  language: string = "en",
  excludedDirs?: string,
  excludedFiles?: string,
  includedDirs?: string,
  includedFiles?: string
): void => {
  if (token !== "") {
    requestBody.token = token;
  }

  // Add provider-based model selection parameters
  requestBody.provider = provider;
  requestBody.model = model;
  if (isCustomModel && customModel) {
    requestBody.custom_model = customModel;
  }

  requestBody.language = language;

  // Add file filter parameters if provided
  if (excludedDirs) {
    requestBody.excluded_dirs = excludedDirs;
  }
  if (excludedFiles) {
    requestBody.excluded_files = excludedFiles;
  }
  if (includedDirs) {
    requestBody.included_dirs = includedDirs;
  }
  if (includedFiles) {
    requestBody.included_files = includedFiles;
  }
};

const createGithubHeaders = (githubToken: string): HeadersInit => {
  const headers: HeadersInit = {
    Accept: "application/vnd.github.v3+json",
  };

  if (githubToken) {
    headers["Authorization"] = `Bearer ${githubToken}`;
  }

  return headers;
};

const createGitlabHeaders = (gitlabToken: string): HeadersInit => {
  const headers: HeadersInit = {
    "Content-Type": "application/json",
  };

  if (gitlabToken) {
    headers["PRIVATE-TOKEN"] = gitlabToken;
  }

  return headers;
};

const createBitbucketHeaders = (bitbucketToken: string): HeadersInit => {
  const headers: HeadersInit = {
    "Content-Type": "application/json",
  };

  if (bitbucketToken) {
    headers["Authorization"] = `Bearer ${bitbucketToken}`;
  }

  return headers;
};

export default function RepoWikiPage() {
  // Get route parameters and search params
  const params = useParams();
  const searchParams = useSearchParams();
  const { addPromptLog } = usePromptLog();

  // Extract owner and repo from route params
  const owner = params.owner as string;
  const repo = params.repo as string;

  // Extract tokens from search params
  const token = searchParams.get("token") || "";
  const localPath = searchParams.get("local_path")
    ? decodeURIComponent(searchParams.get("local_path") || "")
    : undefined;
  const repoUrl = searchParams.get("repo_url")
    ? decodeURIComponent(searchParams.get("repo_url") || "")
    : undefined;
  const providerParam = searchParams.get("provider") || "";
  const modelParam = searchParams.get("model") || "";
  const isCustomModelParam = searchParams.get("is_custom_model") === "true";
  const customModelParam = searchParams.get("custom_model") || "";
  const language = searchParams.get("language") || "en";
  const repoType = repoUrl?.includes("bitbucket.org")
    ? "bitbucket"
    : repoUrl?.includes("gitlab.com")
    ? "gitlab"
    : repoUrl?.includes("github.com")
    ? "github"
    : searchParams.get("type") || "github";

  // Import language context for translations
  const { messages } = useLanguage();

  // Initialize repo info
  const repoInfo = useMemo<RepoInfo>(
    () => ({
      owner,
      repo,
      type: repoType,
      token: token || null,
      localPath: localPath || null,
      repoUrl: repoUrl || null,
    }),
    [owner, repo, repoType, localPath, repoUrl, token]
  );

  type RagQueryDocs = {
    rag_query: string;
    docs: { file_path: string; score: number; text: string }[];
  };

  // State variables
  const [retrievedDocs, setRetrievedDocs] = useState<RagQueryDocs[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState<string | undefined>(
    messages.loading?.initializing || "Initializing wiki generation..."
  );
  const [error, setError] = useState<string | null>(null);
  const [wikiStructure, setWikiStructure] = useState<
    WikiStructure | undefined
  >();
  const [currentPageId, setCurrentPageId] = useState<string | undefined>();
  const [generatedPages, setGeneratedPages] = useState<
    Record<string, WikiPage>
  >({});
  const [pagesInProgress, setPagesInProgress] = useState(new Set<string>());
  const [isExporting, setIsExporting] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [originalMarkdown, setOriginalMarkdown] = useState<
    Record<string, string>
  >({});
  const [requestInProgress, setRequestInProgress] = useState(false);
  const [currentToken, setCurrentToken] = useState(token); // Track current effective token
  const [effectiveRepoInfo, setEffectiveRepoInfo] = useState(repoInfo); // Track effective repo info with cached data
  const [embeddingError, setEmbeddingError] = useState(false);
  const [textReceivedCount, setTextReceivedCount] = useState(0);
  const [structureStartTime, setStructureStartTime] = useState<number | null>(
    null
  );
  const [elapsedTime, setElapsedTime] = useState(0);
  const [pageTextReceivedCount, setPageTextReceivedCount] = useState(0);
  const [pageStartTime, setPageStartTime] = useState<number | null>(null);
  const [pageElapsedTime, setPageElapsedTime] = useState(0);
  const [currentGeneratingPageId, setCurrentGeneratingPageId] = useState<
    string | null
  >(null);
  const [refreshPageIdQueued, setRefreshPageIdQueued] = useState(null);
  const [showWikiTypeInModal, setShowWikiTypeInModal] = useState(true);

  // Control Page Refresh Model Configuation Modal
  const [pendingPageRefreshParams, setPendingPageRefreshParams] =
    useState<ModelSelectionParams | null>(null);

  // Control override enable and disable of prompt editing flow
  const [enablePromptEditing, setEnablePromptEditing] = useState(true); // Default = enabled

  // Control prompt editing widget
  const [showPromptModal, setShowPromptModal] = useState(false);
  const [promptToEdit, setPromptToEdit] = useState("");
  const [modelToShow, setModelToShow] = useState("");
  const [onPromptConfirm, setOnPromptConfirm] = useState<
    null | ((editedPrompt: string) => void)
  >(null);
  const [onPromptCancel, setOnPromptCancel] = useState<null | (() => void)>(
    null
  );
  const [promptModalTitle, setPromptModalTitle] = useState(
    "Edit Generator Prompt"
  );

  // const [pendingPrompt, setPendingPrompt] = useState('');
  const [pendingPageId, setPendingPageId] = useState<string | null>(null);

  // Analytics state
  const [wikiAnalytics, setWikiAnalytics] = useState<{
    model: string;
    provider: string;
    tokensReceived: number;
    timeTaken: number;
  } | null>(null);

  const [pageAnalytics, setPageAnalytics] = useState<
    Record<
      string,
      {
        model: string;
        provider: string;
        tokensReceived: number;
        timeTaken: number;
      }
    >
  >({});

  // Model selection state variables
  const [selectedProviderState, setSelectedProviderState] =
    useState(providerParam);
  const [selectedModelState, setSelectedModelState] = useState(modelParam);
  const [isCustomSelectedModelState, setIsCustomSelectedModelState] =
    useState(isCustomModelParam);
  const [customSelectedModelState, setCustomSelectedModelState] =
    useState(customModelParam);
  const [showModelOptions, setShowModelOptions] = useState(false); // Controls whether to show model options
  const excludedDirs = searchParams.get("excluded_dirs") || "";
  const excludedFiles = searchParams.get("excluded_files") || "";
  const [modelExcludedDirs, setModelExcludedDirs] = useState(excludedDirs);
  const [modelExcludedFiles, setModelExcludedFiles] = useState(excludedFiles);
  const includedDirs = searchParams.get("included_dirs") || "";
  const includedFiles = searchParams.get("included_files") || "";
  const [modelIncludedDirs, setModelIncludedDirs] = useState(includedDirs);
  const [modelIncludedFiles, setModelIncludedFiles] = useState(includedFiles);

  // Wiki type state - default to comprehensive view
  const isComprehensiveParam: boolean =
    searchParams.get("comprehensive") !== "false";
  const [isComprehensiveView, setIsComprehensiveView] =
    useState(isComprehensiveParam);
  // Using useRef for activeContentRequests to maintain a single instance across renders
  // This map tracks which pages are currently being processed to prevent duplicate requests
  // Note: In a multi-threaded environment, additional synchronization would be needed,
  // but in React's single-threaded model, this is safe as long as we set the flag before any async operations
  const activeContentRequests = useRef(new Map<string, boolean>()).current;
  const [structureRequestInProgress, setStructureRequestInProgress] =
    useState(false);
  // Create a flag to track if data was loaded from cache to prevent immediate re-save
  const cacheLoadedSuccessfully = useRef(false);

  // Create a flag to ensure the effect only runs once
  const effectRan = React.useRef(false);

  // State for Ask modal
  const [isAskModalOpen, setIsAskModalOpen] = useState(false);
  const askComponentRef = useRef<{ clearConversation: () => void } | null>(
    null
  );

  // Authentication state
  const [authRequired, setAuthRequired] = useState<boolean>(false);
  const [authCode, setAuthCode] = useState<string>("");
  const [isAuthLoading, setIsAuthLoading] = useState<boolean>(true);

  // Default branch state
  const [defaultBranch, setDefaultBranch] = useState<string>("main");

  // Deep Research state for page refresh
  const [deepResearchEnabled, setDeepResearchEnabled] = useState(false);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [maxIterations, setMaxIterations] = useState(5);
  const [pageConversationHistory, setPageConversationHistory] = useState<
    Record<string, Message[]>
  >({});
  const [researchComplete, setResearchComplete] = useState(false);
  const [embeddingProgress, setEmbeddingProgress] = useState<string | null>(
    null
  );

  // WebSocket reference
  const webSocketRef = useRef<WebSocket | null>(null);
  const conversationHistoryRef = useRef(pageConversationHistory);
  const currentIterationRef = useRef(currentIteration);
  const researchCompleteRef = useRef(researchComplete);
  let latestIterationRef = useRef(0);
  const generatedPagesRef = useRef(generatedPages);

  function showPromptEditModal(
    prompt: string,
    model: string = "-",
    title: string = "Edit Generation Prompt"
  ): Promise<string> {
    return new Promise((resolve, reject) => {
      setPromptToEdit(prompt);
      setModelToShow(model);
      setPromptModalTitle(title);
      setShowPromptModal(true);
      setOnPromptConfirm(() => (editedPrompt: string) => {
        resolve(editedPrompt);
      });
      setOnPromptCancel(() => () => {
        reject(new Error("Prompt editing cancelled"));
      });
    });
  }

  // Helper function to generate proper repository file URLs
  const generateFileUrl = useCallback(
    (filePath: string): string => {
      if (effectiveRepoInfo.type === "local") {
        // For local repositories, we can't generate web URLs
        return filePath;
      }

      const repoUrl = effectiveRepoInfo.repoUrl;
      if (!repoUrl) {
        return filePath;
      }

      try {
        const url = new URL(repoUrl);
        const hostname = url.hostname;

        if (hostname === "github.com" || hostname.includes("github")) {
          // GitHub URL format: https://github.com/owner/repo/blob/branch/path
          return `${repoUrl}/blob/${defaultBranch}/${filePath}`;
        } else if (hostname === "gitlab.com" || hostname.includes("gitlab")) {
          // GitLab URL format: https://gitlab.com/owner/repo/-/blob/branch/path
          return `${repoUrl}/-/blob/${defaultBranch}/${filePath}`;
        } else if (
          hostname === "bitbucket.org" ||
          hostname.includes("bitbucket")
        ) {
          // Bitbucket URL format: https://bitbucket.org/owner/repo/src/branch/path
          return `${repoUrl}/src/${defaultBranch}/${filePath}`;
        }
      } catch (error) {
        console.warn("Error generating file URL:", error);
      }

      // Fallback to just the file path
      return filePath;
    },
    [effectiveRepoInfo, defaultBranch]
  );

  // Memoize repo info to avoid triggering updates in callbacks

  // Add useEffect to handle scroll reset
  useEffect(() => {
    // Scroll to top when currentPageId changes
    const wikiContent = document.getElementById("wiki-content");
    if (wikiContent) {
      wikiContent.scrollTo({ top: 0, behavior: "smooth" });
    }
  }, [currentPageId]);

  useEffect(() => {
    currentIterationRef.current = currentIteration;
  }, [currentIteration]);

  useEffect(() => {
    researchCompleteRef.current = researchComplete;
  }, [researchComplete]);

  // Update conversationHistoryRef whenever pageConversationHistory changes
  useEffect(() => {
    conversationHistoryRef.current = pageConversationHistory;
  }, [pageConversationHistory]);

  useEffect(() => {
    latestIterationRef.current = currentIteration;
  }, [currentIteration]);

  useEffect(() => {
    generatedPagesRef.current = generatedPages;
  }, [generatedPages]);

  // close the modal when escape is pressed
  useEffect(() => {
    const handleEsc = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsAskModalOpen(false);
      }
    };

    if (isAskModalOpen) {
      window.addEventListener("keydown", handleEsc);
    }

    // Cleanup on unmount or when modal closes
    return () => {
      window.removeEventListener("keydown", handleEsc);
    };
  }, [isAskModalOpen]);

  // Fetch authentication status on component mount
  useEffect(() => {
    const fetchAuthStatus = async () => {
      try {
        setIsAuthLoading(true);
        const response = await fetch("/api/auth/status");
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setAuthRequired(data.auth_required);
      } catch (err) {
        console.error("Failed to fetch auth status:", err);
        // Assuming auth is required if fetch fails to avoid blocking UI for safety
        setAuthRequired(true);
      } finally {
        setIsAuthLoading(false);
      }
    };

    fetchAuthStatus();
  }, []);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (structureStartTime && structureRequestInProgress) {
      interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - structureStartTime) / 1000));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [structureStartTime, structureRequestInProgress]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (pageStartTime && currentGeneratingPageId) {
      interval = setInterval(() => {
        setPageElapsedTime(Math.floor((Date.now() - pageStartTime) / 1000));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [pageStartTime, currentGeneratingPageId]);

  function isValidReadmeAnalysis(obj: any): obj is ReadmeAnalysis {
    return (
      obj &&
      typeof obj.project_purpose === "string" &&
      Array.isArray(obj.key_features) &&
      typeof obj.architecture_overview === "string" &&
      Array.isArray(obj.main_technologies) &&
      Array.isArray(obj.important_sections)
    );
  }

  const analyzeReadmeContent = useCallback(
    async (
      readmeString: string,
      owner: string,
      repo: string
    ): Promise<ReadmeContent> => {
      let readmeAnalysisPrompt = `Analyze this README.md file and extract key information:  
    
  <readme>  
  ${readmeString}  
  </readme>  
    
  Extract and return in JSON format:  
  {  
    "project_purpose": "Brief description of what the project does",  
    "key_features": ["feature1", "feature2", ...],  
    "architecture_overview": "High-level architecture description",  
    "main_technologies": ["tech1", "tech2", ...],  
    "important_sections": ["section1", "section2", ...]  
  }  
    
  Return ONLY valid JSON, no markdown formatting.`;

      if (enablePromptEditing) {
        try {
          // Update the prompt if edited
          const model_to_use = `${selectedProviderState}/${
            isCustomSelectedModelState
              ? customSelectedModelState
              : selectedModelState
          }`;
          readmeAnalysisPrompt = await showPromptEditModal(
            readmeAnalysisPrompt,
            model_to_use,
            "Readme Analysis and Summarization Prompt"
          );
        } catch (err) {
          console.error("Error in editing Readme Analysis prompt", err);
        }
      }

      const requestBody: ChatCompletionRequest = {
        repo_url: getRepoUrl(effectiveRepoInfo),
        type: effectiveRepoInfo.type,
        messages: [
          {
            role: "user",
            content: readmeAnalysisPrompt,
          },
        ],
      };

      addTokensToRequestBody(
        requestBody,
        currentToken,
        effectiveRepoInfo.type,
        selectedProviderState,
        selectedModelState,
        isCustomSelectedModelState,
        customSelectedModelState,
        language,
        modelExcludedDirs,
        modelExcludedFiles,
        modelIncludedDirs,
        modelIncludedFiles
      );

      const requestStartTime = Date.now();
      try {
        const response = await fetch(`/api/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`README analysis failed: ${response.status}`);
        }

        let responseText = "";
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) throw new Error("Failed to get response reader");

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          responseText += decoder.decode(value, { stream: true });
        }

        addPromptLog({
          source: "ReadmeSummarize",
          prompt: requestBody.messages
            .map((m) => `${m.role}: ${m.content}`)
            .join("\n\n"),
          response: responseText,
          timestamp: Date.now(),
          model: `${selectedProviderState}/${
            isCustomSelectedModelState
              ? customSelectedModelState
              : selectedModelState
          }`,
          timeTaken: (Date.now() - requestStartTime) / 1000,
        });

        // Extract JSON from response (may be wrapped in markdown)
        const jsonMatch = responseText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const parsed = JSON.parse(jsonMatch[0]);
          if (isValidReadmeAnalysis(parsed)) {
            return parsed;
          } else {
            console.warn("Parsed README analysis is not valid:", parsed);
          }
        }

        throw new Error("Could not parse README analysis response");
      } catch (error) {
        console.error("Error analyzing README:", error);
      }
      // If analysis fails, return original README
      return readmeString;
    },
    [
      currentToken,
      effectiveRepoInfo,
      selectedProviderState,
      selectedModelState,
      isCustomSelectedModelState,
      customSelectedModelState,
      language,
      modelExcludedDirs,
      modelExcludedFiles,
      modelIncludedDirs,
      modelIncludedFiles,
    ]
  );

  function isValidRepoDeepAnalysis(obj: any): obj is RepoDeepAnalysis {
    return (
      obj &&
      typeof obj.domain === "string" &&
      typeof obj.architecture_style === "string" &&
      Array.isArray(obj.key_subsystems) &&
      Array.isArray(obj.data_flow_patterns) &&
      Array.isArray(obj.integration_points) &&
      Array.isArray(obj.core_abstractions) &&
      Array.isArray(obj.documentation_priorities) &&
      typeof obj.domain_concepts === "object"
    );
  }

  const performDeepCodebaseAnalysis = useCallback(
    async (
      fileTreeString: string,
      readme: ReadmeContent,
      basicAnalysis: RepoBasicAnalysis,
      owner: string,
      repo: string
    ): Promise<RepoDeepAnalysis | null> => {
      const readmeText =
        typeof readme === "string" ? readme : readme.project_purpose;

      let deepAnalysisPrompt = `You are analyzing the ${owner}/${repo} codebase to understand its core architecture and domain.  
  
BASIC ANALYSIS (from heuristics):  
- Primary Language: ${basicAnalysis.primaryLanguage}  
- Framework: ${basicAnalysis.framework}  
- Architecture: ${basicAnalysis.architecturePattern}  
- Complexity: ${basicAnalysis.complexityScore}/10  
  
<file_tree>  
${fileTreeString}  
</file_tree>  
  
<readme>  
${readmeText}  
</readme>  
  
Perform DEEP ANALYSIS and return JSON:  
  
{  
  "domain": "What problem domain? (e.g., 'web scraping', 'ML pipeline', 'e-commerce')",  
  "architecture_style": "Primary pattern (e.g., 'microservices', 'event-driven', 'layered')",  
  "key_subsystems": [  
    {  
      "name": "Subsystem name",  
      "purpose": "What it does",  
      "entry_points": ["main files"],  
      "dependencies": ["what it depends on"]  
    }  
  ],  
  "data_flow_patterns": ["How data moves through the system"],  
  "integration_points": ["External services, APIs, databases"],  
  "core_abstractions": ["Key classes/interfaces central to understanding"],  
  "documentation_priorities": ["What aspects need most documentation"],  
  "domain_concepts": {  
    "concept_name": "explanation of domain-specific concept"  
  }  
}  
  
Return ONLY valid JSON, no markdown formatting.`;

      if (enablePromptEditing) {
        try {
          const model_to_use = `${selectedProviderState}/${
            isCustomSelectedModelState
              ? customSelectedModelState
              : selectedModelState
          }`;
          deepAnalysisPrompt = await showPromptEditModal(
            deepAnalysisPrompt,
            model_to_use,
            "Deep Codebase Analysis Prompt"
          );
        } catch (err) {
          console.error("Error editing deep analysis prompt", err);
          throw err;
        }
      }

      const requestBody: ChatCompletionRequest = {
        repo_url: getRepoUrl(effectiveRepoInfo),
        type: effectiveRepoInfo.type,
        messages: [
          {
            role: "user",
            content: deepAnalysisPrompt,
          },
        ],
      };

      addTokensToRequestBody(
        requestBody,
        currentToken,
        effectiveRepoInfo.type,
        selectedProviderState,
        selectedModelState,
        isCustomSelectedModelState,
        customSelectedModelState,
        language,
        modelExcludedDirs,
        modelExcludedFiles,
        modelIncludedDirs,
        modelIncludedFiles
      );

      const requestStartTime = Date.now();
      try {
        /*
        const response = await fetch(`/api/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`Deep analysis failed: ${response.status}`);
        }

        let responseText = "";
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) throw new Error("Failed to get response reader");

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          responseText += decoder.decode(value, { stream: true });
        }
        */
        // Switch to WebSocket-based streaming
        let responseText = await new Promise<string>((resolve, reject) => {
          let accumulatedText = "";

          createChatWebSocket(
            requestBody,
            (message: string) => {
              try {
                const parsed = JSON.parse(message);
                if (parsed.type === "progress") {
                  setLoadingMessage(parsed.message);
                  if (
                    parsed.message.includes("embed") ||
                    parsed.message.includes("Embedding")
                  ) {
                    setEmbeddingProgress(parsed.message);
                  }
                  return; // Don't add to response text
                }
              } catch {
                // Not JSON, treat as regular content
              }
              accumulatedText += message;
            },
            (error: Event) => {
              console.error("WebSocket error in README analysis:", error);
              reject(error);
            },
            () => {
              resolve(accumulatedText);
            }
          );
        });

        addPromptLog({
          source: "DeepAnalysis",
          prompt: requestBody.messages
            .map((m) => `${m.role}: ${m.content}`)
            .join("\n\n"),
          response: responseText,
          timestamp: Date.now(),
          model: `${selectedProviderState}/${
            isCustomSelectedModelState
              ? customSelectedModelState
              : selectedModelState
          }`,
          timeTaken: (Date.now() - requestStartTime) / 1000,
        });

        const jsonMatch = responseText.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          try {
            const parsed = JSON.parse(jsonMatch[0]);

            // Validate the structure matches RepoDeepAnalysis
            if (isValidRepoDeepAnalysis(parsed)) {
              return parsed;
            } else {
              console.warn(
                "Deep analysis JSON structure validation failed:",
                parsed
              );
              return null;
            }
          } catch (parseError) {
            console.error("Failed to parse deep analysis JSON:", parseError);
            return null;
          }
        }

        throw new Error("Could not parse deep analysis response");
      } catch (error) {
        console.error("Error in deep analysis:", error);
      }

      return null;
    },
    [
      currentToken,
      effectiveRepoInfo,
      selectedProviderState,
      selectedModelState,
      isCustomSelectedModelState,
      customSelectedModelState,
      language,
      modelExcludedDirs,
      modelExcludedFiles,
      modelIncludedDirs,
      modelIncludedFiles,
      enablePromptEditing,
      showPromptEditModal,
      addPromptLog,
      getRepoUrl,
      addTokensToRequestBody,
    ]
  );

  const buildStructureGenerationPrompt = async (
    fileTree: string,
    readme: ReadmeContent,
    owner: string,
    repo: string,
    isComprehensiveView: boolean,
    basicAnalysis: RepoBasicAnalysis,
    deepAnalysis: RepoDeepAnalysis | null
  ) => {
    // Format README content
    let readmeSection: string;
    if (typeof readme === "string") {
      readmeSection = `2. The README file of the project:  
<readme>  
${readme}  
</readme>`;
    } else {
      readmeSection = `2. README Analysis Summary:  
- Project Purpose: ${readme.project_purpose}  
- Key Features: ${readme.key_features.join(", ")}  
- Architecture: ${readme.architecture_overview}  
- Technologies: ${readme.main_technologies.join(", ")}  
- Important Sections: ${readme.important_sections.join(", ")}`;
    }

    // Add deep analysis section if available
    let deepAnalysisSection = "";
    if (deepAnalysis) {
      deepAnalysisSection = `  
  
3. Deep Codebase Analysis:  
    - Domain: ${deepAnalysis.domain}  
    - Architecture Style: ${deepAnalysis.architecture_style}  
    - Key Subsystems: ${deepAnalysis.key_subsystems
      .map((s: any) => s.name)
      .join(", ")}  
    - Data Flow: ${deepAnalysis.data_flow_patterns.join("; ")}  
    - Core Abstractions: ${deepAnalysis.core_abstractions.join(", ")}  
  
Domain-Specific Concepts:  
${Object.entries(deepAnalysis.domain_concepts || {})
  .map(([k, v]) => `- ${k}: ${v}`)
  .join("\n")}  
  
Documentation Priorities:  
${deepAnalysis.documentation_priorities.join("\n- ")}`;
    }

    const promptContent = `Analyze this ${
      basicAnalysis.type
    } repository ${owner}/${repo} and create a DOMAIN-AWARE wiki structure.  
  
Repository Analysis:  
- Primary Language: ${basicAnalysis.primaryLanguage}  
- Framework: ${basicAnalysis.framework}  
- Architecture Pattern: ${basicAnalysis.architecturePattern}  
- Complexity Score: ${basicAnalysis.complexityScore}/10  
  
Based on this analysis, suggest a ${
      basicAnalysis.complexityScore > 7 ? "comprehensive" : "concise"
    } wiki structure.  
  
1. The complete file tree of the project:  
<file_tree>  
${fileTree}  
</file_tree>  
  
${readmeSection}  
${deepAnalysisSection}  
  
CRITICAL: Use the domain analysis to create wiki pages that reflect the ACTUAL problem domain and architecture patterns of this codebase, not generic software documentation.  
  
For example:  
- If domain is "web scraping", create pages like "Target Site Analysis", "Data Extraction Pipeline", "Anti-Detection Mechanisms"  
- If domain is "ML pipeline", create pages like "Data Ingestion", "Feature Engineering", "Model Training", "Evaluation Metrics"  
- If domain is "e-commerce", create pages like "Product Catalog", "Shopping Cart", "Payment Processing", "Order Fulfillment"  
  
When designing the wiki structure, classify each page with an appropriate page_type:  
- **architecture**: System design, component relationships, architectural patterns  
- **api**: API endpoints, request/response formats, authentication  
- **configuration**: Configuration files, environment variables, settings  
- **deployment**: Deployment procedures, infrastructure, CI/CD  
- **data_model**: Database schemas, data structures, entity relationships  
- **component**: UI components, modules, widgets (frontend or backend)  
- **general**: Overview, getting started, general documentation  
  
${
  isComprehensiveView
    ? `  
Create a structured wiki with sections that reflect the DOMAIN and ARCHITECTURE:  
${
  deepAnalysis
    ? `  
Based on the deep analysis, organize sections around:  
${deepAnalysis.key_subsystems
  .map((s: any) => `- ${s.name}: ${s.purpose}`)
  .join("\n")}  
`
    : `  
- Overview (general information about the project)  
- System Architecture (how the system is designed)  
- Core Features (key functionality)  
- Data Management/Flow  
- Frontend Components (if applicable)  
- Backend Systems (server-side components)  
- Deployment/Infrastructure  
`
}  
  
Return your analysis in the following XML format:  
<wiki_structure>  
  <title>[Overall title for the wiki]</title>  
  <description>[Brief description of the repository]</description>  
  <sections>  
    <section id="section-1">  
      <title>[Section title reflecting domain/subsystem]</title>  
      <pages>  
        <page_ref>page-1</page_ref>  
      </pages>  
    </section>  
  </sections>  
  <pages>  
    <page id="page-1">  
      <title>[Domain-specific page title]</title>  
      <description>[Brief description]</description>  
      <page_type>architecture|api|configuration|deployment|data_model|component|general</page_type>  
      <importance>high|medium|low</importance>  
      <relevant_files>  
        <file_path>[Path to a relevant file]</file_path>  
      </relevant_files>  
      <related_pages>  
        <related>page-2</related>  
      </related_pages>  
      <parent_section>section-1</parent_section>  
    </page>  
  </pages>  
</wiki_structure>  
`
    : `[... concise format ...]`
}  
  
IMPORTANT FORMATTING INSTRUCTIONS:  
- Return ONLY the valid XML structure specified above  
- DO NOT wrap the XML in markdown code blocks  
- Start directly with <wiki_structure> and end with </wiki_structure>  
  
IMPORTANT:  
1. Create ${
      isComprehensiveView ? "8-12" : "4-6"
    } pages that reflect the DOMAIN and ARCHITECTURE  
2. Page titles should use domain-specific terminology from the deep analysis  
3. The relevant_files should be actual files from the repository  
4. Return ONLY valid XML with no markdown delimiters`;

    return promptContent;
  };

  const old_buildStructureGenerationPrompt = async (
    fileTree: string,
    readme: ReadmeContent,
    owner: string,
    repo: string,
    isComprehensiveView: boolean
  ) => {
    const repoAnalysis = await analyzeRepository(
      fileTree,
      typeof readme === "string" ? readme : readme.project_purpose
    );
    // Format README content
    let readmeSection: string;
    if (typeof readme === "string") {
      readmeSection = `2. The README file of the project:  
<readme>  
${readme}  
</readme>`;
    } else {
      readmeSection = `2. README Analysis Summary:  
- Project Purpose: ${readme.project_purpose}  
- Key Features: ${readme.key_features.join(", ")}  
- Architecture: ${readme.architecture_overview}  
- Technologies: ${readme.main_technologies.join(", ")}  
- Important Sections: ${readme.important_sections.join(", ")}`;
    }
    const promptContent = `Analyze this ${
      repoAnalysis.type
    } repository ${owner}/${repo} and create a wiki structure for it.
  
Repository Analysis:  
- Primary Language: ${repoAnalysis.primaryLanguage}  
- Framework: ${repoAnalysis.framework}  
- Architecture Pattern: ${repoAnalysis.architecturePattern}  
- Complexity Score: ${repoAnalysis.complexityScore}/10  
  
Based on this analysis, suggest a ${
      repoAnalysis.complexityScore > 7 ? "comprehensive" : "concise"
    } wiki structure.  
  
1. The complete file tree of the project:  
<file_tree>  
${fileTree}  
</file_tree>  

2. The README file of the project:
<readme>
${readmeSection}
</readme>

I want to create a wiki for this repository. Determine the most logical structure for a wiki based on the repository's content.

IMPORTANT: The wiki content will be generated in ${
      language === "en"
        ? "English"
        : language === "ja"
        ? "Japanese (日本語)"
        : language === "zh"
        ? "Mandarin Chinese (中文)"
        : language === "zh-tw"
        ? "Traditional Chinese (繁體中文)"
        : language === "es"
        ? "Spanish (Español)"
        : language === "kr"
        ? "Korean (한国語)"
        : language === "vi"
        ? "Vietnamese (Tiếng Việt)"
        : language === "pt-br"
        ? "Brazilian Portuguese (Português Brasileiro)"
        : language === "fr"
        ? "Français (French)"
        : language === "ru"
        ? "Русский (Russian)"
        : "English"
    } language.

When designing the wiki structure, include pages that would benefit from visual diagrams, such as:
- Architecture overviews
- Data flow descriptions
- Component relationships
- Process workflows
- State machines
- Class hierarchies

When designing the wiki structure, classify each page with an appropriate page_type:  
- **architecture**: System design, component relationships, architectural patterns  
- **api**: API endpoints, request/response formats, authentication  
- **configuration**: Configuration files, environment variables, settings  
- **deployment**: Deployment procedures, infrastructure, CI/CD  
- **data_model**: Database schemas, data structures, entity relationships  
- **component**: UI components, modules, widgets (frontend or backend)  
- **general**: Overview, getting started, general documentation

${
  isComprehensiveView
    ? `
Create a structured wiki with the following main sections:
- Overview (general information about the project)
- System Architecture (how the system is designed)
- Core Features (key functionality)
- Data Management/Flow: If applicable, how data is stored, processed, accessed, and managed (e.g., database schema, data pipelines, state management).
- Frontend Components (UI elements, if applicable.)
- Backend Systems (server-side components)
- Model Integration (AI model connections)
- Deployment/Infrastructure (how to deploy, what's the infrastructure like)
- Extensibility and Customization: If the project architecture supports it, explain how to extend or customize its functionality (e.g., plugins, theming, custom modules, hooks).

Each section should contain relevant pages. For example, the "Frontend Components" section might include pages for "Home Page", "Repository Wiki Page", "Ask Component", etc.

Return your analysis in the following XML format:

<wiki_structure>
  <title>[Overall title for the wiki]</title>
  <description>[Brief description of the repository]</description>
  <sections>
    <section id="section-1">
      <title>[Section title]</title>
      <pages>
        <page_ref>page-1</page_ref>
        <page_ref>page-2</page_ref>
      </pages>
      <subsections>
        <section_ref>section-2</section_ref>
      </subsections>
    </section>
    <!-- More sections as needed -->
  </sections>
  <pages>
    <page id="page-1">
      <title>[Page title]</title>
      <description>[Brief description of what this page will cover]</description>
      <page_type>architecture|api|configuration|deployment|data_model|component|general</page_type>
      <importance>high|medium|low</importance>
      <relevant_files>
        <file_path>[Path to a relevant file]</file_path>
        <!-- More file paths as needed -->
      </relevant_files>
      <related_pages>
        <related>page-2</related>
        <!-- More related page IDs as needed -->
      </related_pages>
      <parent_section>section-1</parent_section>
    </page>
    <!-- More pages as needed -->
  </pages>
</wiki_structure>
`
    : `
Return your analysis in the following XML format:

<wiki_structure>
  <title>[Overall title for the wiki]</title>
  <description>[Brief description of the repository]</description>
  <pages>
    <page id="page-1">
      <title>[Page title]</title>
      <description>[Brief description of what this page will cover]</description>
      <page_type>architecture|api|configuration|deployment|data_model|component|general</page_type>
      <importance>high|medium|low</importance>
      <relevant_files>
        <file_path>[Path to a relevant file]</file_path>
        <!-- More file paths as needed -->
      </relevant_files>
      <related_pages>
        <related>page-2</related>
        <!-- More related page IDs as needed -->
      </related_pages>
    </page>
    <!-- More pages as needed -->
  </pages>
</wiki_structure>
`
}

IMPORTANT FORMATTING INSTRUCTIONS:
- Return ONLY the valid XML structure specified above
- DO NOT wrap the XML in markdown code blocks (no \`\`\` or \`\`\`xml)
- DO NOT include any explanation text before or after the XML
- Ensure the XML is properly formatted and valid
- Start directly with <wiki_structure> and end with </wiki_structure>

IMPORTANT:
1. Create ${isComprehensiveView ? "8-12" : "4-6"} pages that would make a ${
      isComprehensiveView ? "comprehensive" : "concise"
    } wiki for this repository
2. Each page should focus on a specific aspect of the codebase (e.g., architecture, key features, setup)
3. The relevant_files should be actual files from the repository that would be used to generate that page
4. Return ONLY valid XML with the structure specified above, with no markdown code block delimiters
5. Ensure that there are no duplicate section ids and page ids`;

    return promptContent;
  }; // end of buildStructureGenerationPrompt

  const getTypeSpecificInstructions = (pageType?: string): string => {
    switch (pageType) {
      case "architecture":
        return `  
ARCHITECTURE PAGE SPECIFIC INSTRUCTIONS:  
- Focus heavily on system design and component relationships  
- Include at least 3 architecture diagrams (system overview, component interaction, data flow)  
- Emphasize design patterns and architectural decisions  
- Explain scalability and performance considerations`;

      case "api":
        return `  
API DOCUMENTATION SPECIFIC INSTRUCTIONS:  
- Create comprehensive tables for all endpoints (method, path, parameters, responses)  
- Include request/response examples for each endpoint  
- Document authentication and authorization requirements  
- Explain error codes and handling`;

      case "configuration":
        return `  
CONFIGURATION PAGE SPECIFIC INSTRUCTIONS:  
- Create detailed tables of all configuration options  
- Include default values, types, and constraints  
- Provide examples for common configuration scenarios`;

      case "deployment":
        return `  
DEPLOYMENT PAGE SPECIFIC INSTRUCTIONS:  
- Provide step-by-step deployment procedures  
- Include infrastructure diagrams  
- Document prerequisites and dependencies`;

      case "data_model":
        return `  
DATA MODEL PAGE SPECIFIC INSTRUCTIONS:  
- Use ER diagrams extensively to show relationships  
- Create comprehensive tables for all entities/models  
- Document field types, constraints, and relationships`;

      case "component":
        return `  
COMPONENT PAGE SPECIFIC INSTRUCTIONS:  
- Focus on component hierarchy and interfaces  
- Include component interaction diagrams  
- Document props, state, and lifecycle`;

      default:
        return "";
    }
  };

  const getLanguageName = (language: string): string => {
    return language === "en"
      ? "English"
      : language === "ja"
      ? "Japanese (日本語)"
      : language === "zh"
      ? "Mandarin Chinese (中文)"
      : language === "zh-tw"
      ? "Traditional Chinese (繁體中文)"
      : language === "es"
      ? "Spanish (Español)"
      : language === "kr"
      ? "Korean (한국어)"
      : language === "vi"
      ? "Vietnamese (Tiếng Việt)"
      : language === "pt-br"
      ? "Brazilian Portuguese (Português Brasileiro)"
      : language === "fr"
      ? "Français (French)"
      : language === "ru"
      ? "Русский (Russian)"
      : "English";
  };

  const buildPageGenerationPrompt = (
    page: WikiPage,
    params?: ModelSelectionParams,
    deep_research: boolean = false
  ) => {
    const filePaths = page.filePaths;
    // Get type-specific instructions based on page_type
    const typeSpecificInstructions = getTypeSpecificInstructions(page.pageType);

    if (deep_research) {
      console.log(
        `Generating Promt for title(${page.title}),type(${
          page.pageType || "unknown"
        })`
      );
      const promptContent = `You are an expert software architect and technical writer engaged in an **iterative deep research** process (Iteration N) to produce a detailed Markdown wiki page about **"${
        page.title
      }"**.

Your goal is to integrate information from multiple source files, correlate their logic, and progressively refine previous findings to achieve a comprehensive technical understanding.
 

${typeSpecificInstructions}  
  
### Deep Research Objectives
- Perform in-depth code analysis of all provided files.
- Identify and explain cross-file relationships, hidden dependencies, and architectural patterns.
- Validate and, if necessary, correct prior assumptions or summaries.
- Fill gaps, elaborate unclear areas, and expand on complex system interactions.
- Preserve previous structure where appropriate; expand it with new insights.
- This iteration must bring additional clarity and precision.

### Relevant Source Files
${filePaths.map((path) => `- [${path}](${generateFileUrl(path)})`).join("\n")}

### Output Requirements
- Begin with \`<details>\` block listing the files (as above), then \`# ${
        page.title
      }\`.
- Use logical sectioning (##, ###) to organize topics.
- Include diagrams where helpful:
  - Use \`graph TD\` or valid \`sequenceDiagram\` syntax.
  - Prefer smaller, accurate diagrams over complex ones.
- Cite filenames and line numbers for every claim.
- Avoid conclusions; this process continues iteratively.
- Write in clear, professional ${getLanguageName(language)}.

At the end, include:
## Research Evaluation
Summarize what this iteration clarified and what areas need further investigation.

Focus on depth, accuracy, and code-grounded insights.
`;

      return promptContent;
    }

    // Create the prompt content - simplified to avoid message dialogs
    const promptContent = `
You are an expert technical writer documenting **"${
      page.title
    }"** for a software project.  
Use the provided source files as your only source of truth.

### Relevant Files
${filePaths.map((path) => `- [${path}](${generateFileUrl(path)})`).join("\n")}

${typeSpecificInstructions}

Start with a \`<details>\` block listing these files, then a top-level heading \`# ${
      page.title
    }\`.

### Write the following sections
1. **Introduction:** Purpose and context.
2. **Architecture and Components:** Logical explanation using H2/H3 sections.
3. **Visuals:** Use valid Mermaid diagrams (\`graph TD\`, \`sequenceDiagram\`) to show relationships.
4. **Tables:** Summarize key parameters or data models.
5. **Code Snippets:** Include short, relevant examples.
6. **Citations:** Every paragraph or diagram must cite relevant file(s) and line numbers.
7. **Conclusion:** Optional brief summary.

Ensure accuracy, clarity, and completeness.  
Write in ${getLanguageName(language)} and maintain professional tone.
Ensure that all content is Markdown compatible.

CRITICAL STARTING INSTRUCTION:
The very first thing on the page MUST be a \`<details>\` block listing ALL the \`[RELEVANT_SOURCE_FILES]\` you used to generate the content. There MUST be AT LEAST 5 source files listed - if fewer were provided, you MUST find additional related files to include.
Format it exactly like this:
<details>
<summary>Relevant source files</summary>

Remember, do not provide any acknowledgements, disclaimers, apologies, or any other preface before the \`<details>\` block. JUST START with the \`<details>\` block.
`;
    return promptContent;
  };

  const highlightContext = (
    content: string,
    match: string,
    contextLength = 30
  ) => {
    const index = content.indexOf(match);
    if (index === -1) return null;

    const start = Math.max(0, index - contextLength);
    const end = Math.min(content.length, index + match.length + contextLength);
    const before = content.slice(start, index);
    const after = content.slice(index + match.length, end);

    // Highlight match with brackets for visibility
    return `${before}[${match}]${after}`;
  };

  const checkIfResearchComplete = (content: string): boolean => {
    // Helper to log match location
    const logMatch = (match: string) => {
      const snippet = highlightContext(content, match);
      if (snippet) {
        console.log(
          `Found phrase: '${match}' at index ${content.indexOf(match)}`
        );
        console.log(`Context: ...${snippet}...`);
      } else {
        console.log(`Match '${match}' found, but could not extract context`);
      }
    };

    // Check for explicit final conclusion markers
    if (content.includes("## Final Conclusion")) {
      logMatch("## Final Conclusion");
      return true;
    }

    // Check for conclusion sections that don't indicate further research
    if (
      (content.includes("## Conclusion") || content.includes("## Summary")) &&
      !content.includes("I will now proceed to") &&
      !content.includes("Next Steps") &&
      !content.includes("next iteration")
    ) {
      const match = content.includes("## Conclusion")
        ? "## Conclusion"
        : "## Summary";
      logMatch(match);
      return true;
    }

    // Check for phrases that explicitly indicate completion
    const phrases = [
      "This concludes our research",
      "This completes our investigation",
      "This concludes the deep research process",
      "Key Findings and Implementation Details",
      "In conclusion,",
    ];

    for (const phrase of phrases) {
      if (content.includes(phrase)) {
        logMatch(phrase);
        return true;
      }
    }

    // Combined check for "Final" + "Conclusion" appearing together
    if (content.includes("Final") && content.includes("Conclusion")) {
      logMatch("Final");
      logMatch("Conclusion");
      return true;
    }

    return false;
  };

  /**
   * Attempts to fetch chat stream data via HTTP POST as a fallback mechanism
   * @param body - The request payload to be sent to the chat stream API
   * @param page - The WikiPage object containing page information
   * @returns Promise<string> - Returns the response text if successful, or error message if failed
   * @throws Will throw an error if the HTTP response is not OK (status >= 400)
   *
   * @example
   * const result = await fallbackToHttp({
   *   message: "Hello"
   * }, wikiPage);
   */
  const fallbackToHttp = async (body: any, page: WikiPage): Promise<string> => {
    try {
      const response = await fetch(`/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errText = await response.text().catch(() => "Unknown error");
        throw new Error(`HTTP ${response.status}: ${errText}`);
      }

      // Read entire response body as text
      const text = await response.text();
      return text; // ✅ return the content
    } catch (err) {
      console.error("HTTP fallback failed:", err);
      return `Error: ${(err as Error).message}`;
    }
  };

  /**
   * Resets and cleans up all state variables associated with a specific page generation process.
   *
   * @param pageId - The unique identifier of the page to cleanup state for
   *
   * This function:
   * - Removes the page from active content requests tracking
   * - Removes the page from the in-progress pages Set
   * - Clears loading message
   * - Resets current generating page ID
   * - Resets page start time
   * - Resets text received counter
   * - Resets elapsed time
   */
  const cleanupPageState = (pageId: string) => {
    activeContentRequests.delete(pageId);
    setPagesInProgress((prev) => {
      const next = new Set(prev);
      next.delete(pageId);
      return next;
    });
    setLoadingMessage(undefined);
    setCurrentGeneratingPageId(null);
    setPageStartTime(null);
    setPageTextReceivedCount(0);
    setPageElapsedTime(0);
  };

  /**
   * Continues the research process for a specific wiki page using WebSocket communication with fallback to HTTP.
   *
   * @param pageId - The unique identifier of the wiki page to continue research on
   * @param currentContent - The current content of the wiki page
   * @param params - Optional model selection parameters for the research
   * @param params.provider - The AI provider to use
   * @param params.model - The AI model to use
   * @param params.isCustomModel - Flag indicating if using a custom model
   * @param params.customModel - Custom model configuration
   *
   * @remarks
   * This function handles the deep research process by:
   * - Checking if research should continue based on enabled state and iteration limits
   * - Managing WebSocket connection for streaming responses
   * - Handling connection errors with HTTP fallback
   * - Processing incoming messages and updating page content
   * - Managing research iterations and completion state
   * - Cleaning up resources after completion
   *
   * The function will automatically continue research until either:
   * - Maximum iterations are reached
   * - Research is marked as complete
   * - An error occurs that cannot be recovered from
   *
   * @throws Will log errors if WebSocket connection fails or if HTTP fallback fails
   *
   * @returns Promise<void>
   */
  const continuePageResearch = useCallback(
    async (
      pageId: string,
      // currentContent: string,
      params: ModelSelectionParams
    ): Promise<void> => {
      return new Promise<void>(async (resolve) => {
        try {
          const currentIter = currentIterationRef.current;
          const isComplete = researchCompleteRef.current;
          const latestHistory = conversationHistoryRef.current[pageId] || [];

          if (
            !deepResearchEnabled ||
            isComplete ||
            currentIter >= maxIterations
          ) {
            console.log(
              `continuePageResearch: Exiting: deepResearchEnabled:${deepResearchEnabled}, isComplete: ${isComplete}, (${currentIter} >= ${maxIterations}): ${
                currentIter >= maxIterations
              }`
            );
            resolve();
            return;
          }

          await new Promise((r) => setTimeout(r, 2000));

          setIsLoading(true);
          setLoadingMessage(undefined);
          // setLoadingMessage("Continuing deep research...");

          const page = wikiStructure?.pages.find((p) => p.id === pageId);
          if (!page) {
            console.error(
              `continuePageResearch: Exiting: Page ${pageId} not Found. deepResearchEnabled:${deepResearchEnabled}, researchComplete: ${isComplete}, currentIteration: ${currentIter}, maxIterations: ${maxIterations}`
            );

            // setIsLoading(false);
            // setLoadingMessage(undefined);
            resolve();
            return;
          }

          const newHistory: Message[] = [
            ...latestHistory,
            { role: "user", content: "[DEEP RESEARCH] Continue the research" },
          ];

          console.log(
            `continuePageResearch: pageId:${pageId}: currentIteration:${currentIter}, pageConversationHistory:${latestHistory.length}, newHistory:${newHistory.length},  `
          );

          const requestBody: ChatCompletionRequest = {
            repo_url: getRepoUrl(effectiveRepoInfo),
            type: effectiveRepoInfo.type,
            messages: newHistory,
            deep_research: true,
            max_iterations: maxIterations,
            verbose_mode: true,
          };

          const finalProvider = params?.provider ?? selectedProviderState;
          const finalModel = params?.model ?? selectedModelState;
          const finalIsCustomModel =
            params?.isCustomModel ?? isCustomSelectedModelState;
          const finalCustomModel =
            params?.customModel ?? customSelectedModelState;

          addTokensToRequestBody(
            requestBody,
            currentToken,
            effectiveRepoInfo.type,
            finalProvider,
            finalModel,
            finalIsCustomModel,
            finalCustomModel,
            language,
            modelExcludedDirs,
            modelExcludedFiles,
            modelIncludedDirs,
            modelIncludedFiles
          );

          let content = "";
          const pageStartTime = Date.now();
          let pageTokenCount = 0;
          let finished = false;

          setCurrentGeneratingPageId(page.id);
          setPageStartTime(pageStartTime);
          setPageTextReceivedCount(0);
          setPageElapsedTime(0);

          const finalize = (
            currentIterationContent: string,
            allIterationsContent: string
          ) => {
            const timeTaken = Math.floor((Date.now() - pageStartTime) / 1000);

            // Log only the current iteration
            addPromptLog({
              source: `DeepResearch-PageContent-it(${currentIterationRef.current})`,
              prompt: requestBody.messages
                .map((m) => `${m.role}: ${m.content}`)
                .join("\n\n"),
              response: currentIterationContent, // Only current iteration
              timestamp: Date.now(),
              model: `${finalProvider}/${
                finalIsCustomModel ? finalCustomModel : finalModel
              }`,
              timeTaken,
            });

            // Get existing iterations or initialize empty array
            const existingPage = generatedPagesRef.current[pageId];
            const existingIterations = existingPage?.iterations || [];

            // Add new iteration
            const newIteration: WikiPageIteration = {
              iteration: currentIterationRef.current,
              content: currentIterationContent,
              timestamp: Date.now(),
              model: finalModel,
              provider: finalProvider,
            };

            console.log(`existingIterations: ${existingIterations.length}`);
            console.log(`newIteration: ${newIteration.iteration}`);

            // Store all iterations in the page
            // setGeneratedPages((prev) => ({
            //   ...prev,
            //   [pageId]: {
            //     ...page,
            //     content: allIterationsContent,
            //     iterations: [...existingIterations, newIteration],
            //   },
            // }));
            setGeneratedPages((prev) => {
              const prevPage = prev[pageId] || {};

              return {
                ...prev,
                [pageId]: {
                  ...prevPage, // <-- keep stored fields
                  content: allIterationsContent,
                  iterations: [...existingIterations, newIteration],
                },
              };
            });

            setPageConversationHistory((prev) => ({
              ...prev,
              [pageId]: [
                ...newHistory,
                { role: "assistant", content: currentIterationContent }, // Current iteration
              ],
            }));

            console.log(
              `Updated pageConversationHistory: ${
                (conversationHistoryRef.current[pageId] || []).length
              }`
            );

            setPageAnalytics((prev) => ({
              ...prev,
              [pageId]: {
                model: finalModel,
                provider: finalProvider,
                tokensReceived: pageTokenCount,
                timeTaken,
              },
            }));

            //setIsLoading(false);
            //setLoadingMessage(undefined);
            cleanupPageState(pageId);
            resolve();
          };

          const handleMessage = (data: string) => {
            try {
              const parsed = JSON.parse(data);
              if (parsed.type === "iteration_status") {
                setCurrentIteration(parsed.current_iteration || 0);
                console.log(
                  `Research iteration ${parsed.current_iteration} ${
                    parsed.status || "in_progress"
                  }`
                );
              } else if (parsed.type === "llm_prompt") {
                let actualPrompt = parsed.content;
              } else if (parsed.type === "rag_details") {
                let retrieved_docs = parsed.results;
                let query = parsed.query;

                setRetrievedDocs((prev) => [
                  ...prev,
                  { rag_query: query, docs: retrieved_docs },
                ]);
              }
              // If JSON, then the recieved data is control data and not to be included in the text
              return;
            } catch {}
            pageTokenCount += data.length;
            setPageTextReceivedCount(pageTokenCount);
            setPageElapsedTime(Math.floor((Date.now() - pageStartTime) / 1000));
            content += data;
          };

          const handleError = async (err: Event) => {
            console.error("WebSocket error, fallback to HTTP:", err);
            if (!finished) {
              finished = true;
              setLoadingMessage("Error: Falling back to HTTP...");
              const fallbackContent =
                (await fallbackToHttp(requestBody, page)) || "";
              finalize(fallbackContent, fallbackContent);
            }
          };

          const handleClose = () => {
            if (finished) return;
            finished = true;

            const latestIter = currentIterationRef.current;
            const isComplete = false; // checkIfResearchComplete(content);
            const forceComplete = latestIter >= maxIterations;

            let allIterationsContent = content; // Default to current iteration content

            // if (forceComplete && !isComplete) {
            //   content +=
            //     "\n\n## Final Conclusion\nAfter multiple iterations, sufficient insights were gathered.";
            // }
            if (forceComplete || isComplete) {
              // Get all previous iterations
              const previousResponses = (
                conversationHistoryRef.current[pageId] || []
              )
                .filter((msg) => msg.role === "assistant")
                .map((msg, idx) => `### Iteration ${idx + 1}\n${msg.content}`)
                .join("\n\n");

              // Add current iteration
              const currentResponse = `### Iteration ${latestIter}\n${content}`;

              // Combine all iterations
              allIterationsContent = previousResponses
                ? `${previousResponses}\n\n${currentResponse}`
                : currentResponse;
            }

            finalize(content, allIterationsContent);

            if (forceComplete || isComplete) {
              setResearchComplete(true);
              setCurrentIteration(0);
              console.log(`Stopping `);
              setIsLoading(false);
              setLoadingMessage(undefined);
            } else {
              // setLoadingMessage("Continuing next research iteration...");
              setTimeout(() => continuePageResearch(pageId, params), 2000);
              resolve();
            }
          };

          await createChatWebSocket(
            requestBody,
            handleMessage,
            handleError,
            handleClose
          );
        } catch (err) {
          console.error("continuePageResearch error:", err);
          setIsLoading(false);
          setLoadingMessage(undefined);
          resolve();
        }
      });
    },
    [
      deepResearchEnabled,
      maxIterations,
      wikiStructure,
      effectiveRepoInfo,
      selectedProviderState,
      selectedModelState,
      isCustomSelectedModelState,
      customSelectedModelState,
      currentToken,
      modelExcludedDirs,
      modelExcludedFiles,
      modelIncludedDirs,
      modelIncludedFiles,
      language,
      fallbackToHttp,
      addPromptLog,
      checkIfResearchComplete,
      setIsLoading,
    ]
  );

  /**
   * Generates content for a wiki page using either WebSocket or HTTP fallback.
   *
   * @param page - The wiki page object to generate content for
   * @param owner - Repository owner name
   * @param repo - Repository name
   * @param params - Optional model selection parameters
   * @param promptOverride - Optional custom prompt to override default
   * @param force - Whether to force regenerate content even if it exists
   * @param deep_research - Whether to perform deep research with multiple iterations
   * @param max_iterations - Maximum number of research iterations
   *
   * @remarks
   * This function handles the generation of wiki page content by:
   * - Setting up WebSocket connection for streaming responses
   * - Managing state for pages in progress
   * - Handling content generation through iterations
   * - Falling back to HTTP if WebSocket fails
   * - Tracking analytics like tokens and time taken
   * - Managing research continuation for deep research mode
   *
   * @throws Will log error if repository info is missing
   *
   * @returns void
   */
  const generatePageContent = useCallback(
    async (
      page: WikiPage,
      owner: string,
      repo: string,
      params?: ModelSelectionParams,
      promptOverride?: string,
      force: boolean = false,
      deep_research: boolean = false,
      max_iterations: number = 5
    ): Promise<void> => {
      return new Promise<void>(async (resolve) => {
        try {
          // Validate repo info
          if (!owner || !repo) {
            console.error(
              "Invalid repository information. Owner and repo name are required."
            );
            resolve();
            return;
          }

          if (!force && generatedPages[page.id]?.content) {
            console.log(`${page.title} already generated, skipping.`);
            resolve();
            return;
          }

          if (activeContentRequests.get(page.id)) {
            console.log(
              `${page.title} already processing, skipping duplicate.`
            );
            resolve();
            return;
          }

          // 🚀 Initial state setup
          activeContentRequests.set(page.id, true);
          setPagesInProgress((prev) => new Set(prev).add(page.id));

          setGeneratedPages((prev) => ({
            ...prev,
            [page.id]: { ...page, content: "Loading...", iterations: [] },
          }));
          setOriginalMarkdown((prev) => ({ ...prev, [page.id]: "" }));

          // ---- Initialize loading state ----
          console.log(`Starting content generation for page: ${page.title}`);
          setLoadingMessage(`Generating content for ${page.title}...`);

          const repoUrl = getRepoUrl(effectiveRepoInfo);
          const promptContent =
            promptOverride ??
            buildPageGenerationPrompt(page, params, deep_research);

          const requestBody: ChatCompletionRequest = {
            repo_url: repoUrl,
            type: effectiveRepoInfo.type,
            messages: [{ role: "user", content: promptContent }],
            deep_research: deep_research,
            max_iterations: max_iterations,
            verbose_mode: true,
          };

          const finalProvider = params?.provider ?? selectedProviderState;
          const finalModel = params?.model ?? selectedModelState;
          const finalIsCustomModel =
            params?.isCustomModel ?? isCustomSelectedModelState;
          const finalCustomModel =
            params?.customModel ?? customSelectedModelState;

          addTokensToRequestBody(
            requestBody,
            currentToken,
            effectiveRepoInfo.type,
            finalProvider,
            finalModel,
            finalIsCustomModel,
            finalCustomModel,
            language,
            modelExcludedDirs,
            modelExcludedFiles,
            modelIncludedDirs,
            modelIncludedFiles
          );

          let content = "";
          let pageTokenCount = 0;
          let latestIteration = latestIterationRef.current;

          const requestStartTime = Date.now();
          let finished = false;

          setCurrentGeneratingPageId(page.id);
          setPageStartTime(requestStartTime);
          setPageTextReceivedCount(0);
          setPageElapsedTime(0);

          const handleMessage = (data: string) => {
            try {
              const parsed = JSON.parse(data);
              if (parsed.type === "iteration_status") {
                latestIteration = parsed.current_iteration || 0;
                latestIterationRef.current = latestIteration;
                setCurrentIteration(latestIteration);
                // setLoadingMessage(
                //   `Research iteration ${latestIteration} in progress...`
                // );
              } else if (parsed.type === "rag_details") {
                let retrieved_docs = parsed.results;
                let query = parsed.query;

                setRetrievedDocs((prev) => [
                  ...prev,
                  { rag_query: query, docs: retrieved_docs },
                ]);
              }
              // If JSON, then the recieved data is control data and not to be included in the text
              return;
            } catch {}
            content += data;
            pageTokenCount += data.length;
            setPageTextReceivedCount((prev) => prev + data.length);
          };

          const finalize = (finalContent: string) => {
            let completedContent = finalContent;

            // Get existing page data
            const existingPage = generatedPages[page.id] || {};
            const existingIterations = Array.isArray(existingPage.iterations)
              ? existingPage.iterations
              : [];

            const iterationNumber =
              latestIterationRef.current > 0
                ? latestIterationRef.current
                : existingIterations.length + 1;

            // Add current iteration
            const newIteration = {
              iteration: iterationNumber,
              content: completedContent, // Store raw iteration content
              timestamp: Date.now(),
              model: finalModel,
              provider: finalProvider,
            };

            setPageConversationHistory((prev) => ({
              ...prev,
              [page.id]: [
                ...(prev[page.id] || []),
                { role: "assistant", content: finalContent },
              ],
            }));

            if (deep_research) {
              const isComplete = false; //checkIfResearchComplete(finalContent);
              const forceComplete = iterationNumber >= max_iterations;

              if (forceComplete && !isComplete) {
                completedContent +=
                  "\n\n## Final Conclusion\nAfter multiple iterations, we’ve reached sufficient insights.";
              }

              setGeneratedPages((prev) => ({
                ...prev,
                [page.id]: {
                  ...prev[page.id],
                  content: completedContent,
                  iterations: [...existingIterations, newIteration],
                },
              }));

              if (forceComplete || isComplete) {
                setResearchComplete(true);
                setCurrentIteration(0);
                //setLoadingMessage(undefined);
                //setIsLoading(false);
              } else {
                setLoadingMessage("Continuing next research iteration...");
                setTimeout(() => {
                  continuePageResearch(page.id, params || {});
                }, 2000);
              }
            } else {
              setCurrentIteration(0);
              setLoadingMessage("Finalizing content...");
              setTimeout(() => {
                //setIsLoading(false);
                //setLoadingMessage(undefined);
              }, 600);

              setGeneratedPages((prev) => ({
                ...prev,
                [page.id]: {
                  ...prev[page.id],
                  content: completedContent,
                },
              }));
            }

            setPageAnalytics((prev) => ({
              ...prev,
              [page.id]: {
                model: finalModel,
                provider: finalProvider,
                tokensReceived: pageTokenCount,
                timeTaken: Math.floor((Date.now() - requestStartTime) / 1000),
              },
            }));

            addPromptLog({
              source: deep_research
                ? `DeepResearch-PageContent-it(${currentIterationRef.current})`
                : "PageContent",
              prompt: requestBody.messages
                .map((m) => `${m.role}: ${m.content}`)
                .join("\n\n"),
              response: completedContent,
              timestamp: Date.now(),
              model: `${finalProvider}/${
                finalIsCustomModel ? finalCustomModel : finalModel
              }`,
              timeTaken: (Date.now() - requestStartTime) / 1000,
            });

            cleanupPageState(page.id);
            resolve();
          };

          const handleError = async (error: Event) => {
            if (finished) return;
            finished = true;
            console.error("WebSocket error:", error);
            setLoadingMessage("Error occurred. Falling back to HTTP...");
            const fallbackContent = await fallbackToHttp(requestBody, page);
            finalize(fallbackContent || "");
          };

          const handleClose = () => {
            if (finished) return;
            finished = true;
            finalize(content);
          };

          try {
            webSocketRef.current = await createChatWebSocket(
              requestBody,
              handleMessage,
              handleError,
              handleClose
            );
          } catch (err) {
            console.error("Failed to create WebSocket:", err);
            const fallbackContent = await fallbackToHttp(requestBody, page);
            finalize(fallbackContent || "");
          }
        } catch (err) {
          console.error("Error in generatePageContent:", err);
          //setIsLoading(false);
          //setLoadingMessage(undefined);
          resolve();
        }
      });
    },
    [
      generatedPages,
      currentToken,
      effectiveRepoInfo,
      selectedProviderState,
      selectedModelState,
      isCustomSelectedModelState,
      customSelectedModelState,
      modelExcludedDirs,
      modelExcludedFiles,
      modelIncludedDirs,
      modelIncludedFiles,
      language,
      continuePageResearch,
      addTokensToRequestBody,
      fallbackToHttp,
      addPromptLog,
      getRepoUrl,
      setGeneratedPages,
      setPagesInProgress,
      setCurrentIteration,
      setResearchComplete,
      setPageAnalytics,
      checkIfResearchComplete,
      pageConversationHistory,
    ]
  );

  const refreshPage = useCallback(
    (pageId: string) => {
      setShowWikiTypeInModal(false);
      setRefreshPageIdQueued(pageId);
      setIsModelSelectionModalOpen(true);
    },
    [deepResearchEnabled, maxIterations]
  );

  const performPageRefresh = useCallback(
    async (pageId: string, params: ModelSelectionParams, prompt: string) => {
      console.log(`performPageRefresh(${pageId}) called`);
      const page = wikiStructure?.pages.find((p) => p.id === pageId);
      if (!page) return;

      // Clear any existing request tracking for this page
      activeContentRequests.delete(pageId);

      // Reset cache loaded flag to allow auto-save after refresh
      cacheLoadedSuccessfully.current = false;

      // Reset deep research state
      setCurrentIteration(0);
      setResearchComplete(false);
      setPageConversationHistory((prev) => ({
        ...prev,
        [pageId]: [
          {
            role: "user",
            content: prompt,
          },
        ],
      }));

      // Clear the existing content
      setGeneratedPages((prev) => ({
        ...prev,
        [pageId]: { ...page, content: "" },
      }));

      // Clear analytics for this page
      setPageAnalytics((prev) => {
        const updated = { ...prev };
        delete updated[pageId];
        return updated;
      });

      setIsLoading(true);
      setLoadingMessage(`Refreshing ${page.title}...`);

      try {
        // Regenerate the page content
        await generatePageContent(
          page,
          effectiveRepoInfo.owner,
          effectiveRepoInfo.repo,
          params,
          prompt,
          true,
          deepResearchEnabled,
          maxIterations
        );
      } finally {
        setIsLoading(false);
        setLoadingMessage(undefined);
      }
    },
    [
      wikiStructure,
      generatePageContent,
      effectiveRepoInfo,
      selectedProviderState,
      selectedModelState,
      enablePromptEditing,
      deepResearchEnabled,
      maxIterations,
      activeContentRequests,
    ]
  );

  const confirmPrompt = useCallback(
    async (pageId: string, params: ModelSelectionParams) => {
      console.log(`refreshPage(${pageId}) called`);
      const page = wikiStructure?.pages.find((p) => p.id === pageId);
      if (!page) return;

      // const provider = params?.provider ?? selectedProviderState;
      // const model = params?.model ?? selectedModelState;
      // ...populate others as needed
      let prompt = buildPageGenerationPrompt(page, params, deepResearchEnabled);

      if (enablePromptEditing) {
        // Show the modal for review/edit-- pause flow!
        // setPendingPrompt(prompt);
        setPendingPageId(pageId);
        setPendingPageRefreshParams(params);

        // Update the prompt if edited
        const model_to_use = `${params.provider}/${
          params.isCustomModel ? params.customModel : params.model
        }`;
        try {
          prompt = await showPromptEditModal(
            prompt,
            model_to_use,
            `${
              deepResearchEnabled ? "Deep Research" : ""
            } Page Content Generation Prompt for ${page.title}`
          );
        } catch (err) {
          console.log(err);
          return;
        }
      }
      return performPageRefresh(pageId, params, prompt);
    },
    [
      wikiStructure,
      generatePageContent,
      effectiveRepoInfo,
      selectedProviderState,
      selectedModelState,
      enablePromptEditing,
      performPageRefresh,
      deepResearchEnabled,
      maxIterations,
    ]
  );

  async function analyzeRepository(
    fileTree: string | Array<{ path: string; size: number; modified: number }>,
    readme: string
  ): Promise<RepoBasicAnalysis> {
    // const files = fileTree.split('\n').filter(f => f.trim());
    const files =
      typeof fileTree === "string"
        ? fileTree.split("\n").filter((f) => f.trim())
        : fileTree.map((item) => item.path);

    // Detect primary language
    const languageExtensions = {
      ".js": "JavaScript",
      ".ts": "TypeScript",
      ".tsx": "TypeScript",
      ".jsx": "JavaScript",
      ".py": "Python",
      ".java": "Java",
      ".cpp": "C++",
      ".c": "C",
      ".cs": "C#",
      ".go": "Go",
      ".rs": "Rust",
      ".php": "PHP",
      ".rb": "Ruby",
    };

    const langCounts: Record<string, number> = {};
    files.forEach((file) => {
      const ext = file.substring(file.lastIndexOf("."));
      const lang = languageExtensions[ext];
      if (lang) {
        langCounts[lang] = (langCounts[lang] || 0) + 1;
      }
    });

    console.log("langCounts:", langCounts);

    const primaryLanguage =
      Object.entries(langCounts).sort(([, a], [, b]) => b - a)[0]?.[0] ||
      "Unknown";

    // Detect framework
    let framework = "Unknown";
    const packageJsonExists = files.some((f) => f.includes("package.json"));
    const requirementsTxtExists = files.some((f) =>
      f.includes("requirements.txt")
    );

    if (packageJsonExists) {
      if (files.some((f) => f.includes("next.config"))) framework = "Next.js";
      else if (
        files.some(
          (f) => f.includes("src/App.tsx") || f.includes("src/App.jsx")
        )
      )
        framework = "React";
      else if (files.some((f) => f.includes("angular.json")))
        framework = "Angular";
      else if (files.some((f) => f.includes("vue.config")))
        framework = "Vue.js";
      else framework = "Node.js";
    } else if (requirementsTxtExists) {
      if (files.some((f) => f.includes("manage.py"))) framework = "Django";
      else if (files.some((f) => f.includes("app.py") || f.includes("main.py")))
        framework = "Flask/FastAPI";
      else framework = "Python";
    }

    // Detect architecture pattern
    let architecturePattern = "Unknown";
    if (
      files.some((f) => f.includes("api/") || f.includes("backend/")) &&
      files.some((f) => f.includes("src/") || f.includes("frontend/"))
    ) {
      architecturePattern = "Full-stack";
    } else if (files.some((f) => f.includes("api/") || f.includes("server/"))) {
      architecturePattern = "Backend API";
    } else if (
      files.some((f) => f.includes("components/") || f.includes("pages/"))
    ) {
      architecturePattern = "Frontend SPA";
    } else if (files.some((f) => f.includes("lib/") || f.includes("src/lib"))) {
      architecturePattern = "Library/Package";
    }

    // Calculate complexity score (1-10)
    let complexityScore = 1;
    const fileCount = files.length;
    const dirDepth = Math.max(...files.map((f) => f.split("/").length - 1));
    const hasTests = files.some(
      (f) => f.includes("test") || f.includes("spec")
    );
    const hasConfig = files.some(
      (f) => f.includes("config") || f.includes(".env")
    );
    const hasDocs = files.some(
      (f) => f.includes("docs/") || f.includes("README")
    );

    // Base complexity on file count
    if (fileCount > 500) complexityScore += 4;
    else if (fileCount > 200) complexityScore += 3;
    else if (fileCount > 50) complexityScore += 2;
    else complexityScore += 1;

    // Add complexity for directory depth
    if (dirDepth > 5) complexityScore += 2;
    else if (dirDepth > 3) complexityScore += 1;

    // Add complexity for additional features
    if (hasTests) complexityScore += 1;
    if (hasConfig) complexityScore += 1;
    if (hasDocs) complexityScore += 1;

    // Cap at 10
    complexityScore = Math.min(complexityScore, 10);

    // Determine repository type
    let type = "Application";
    if (
      framework.includes("Library") ||
      architecturePattern === "Library/Package"
    ) {
      type = "Library";
    } else if (
      readme.toLowerCase().includes("api") ||
      architecturePattern === "Backend API"
    ) {
      type = "API Service";
    } else if (architecturePattern === "Full-stack") {
      type = "Full-stack Application";
    }

    return {
      type,
      primaryLanguage,
      framework,
      architecturePattern,
      complexityScore,
    };
  }

  /**
  // Determine the wiki structure from repository data
  // This multi-step approach performs:
  // Basic heuristic analysis (analyzeRepository) - fast, deterministic metadata extraction
  // README summarization (analyzeReadmeContent) - extracts key information from documentation
  // Deep domain profiling (performDeepCodebaseAnalysis) - LLM-powered understanding of problem domain, architecture patterns, and core abstractions
  // Domain-aware structure generation - uses all previous analysis to create wiki structure that reflects the actual codebase's purpose and design
  */
  const determineWikiStructure = useCallback(
    async (
      fileTree:
        | string
        | Array<{ path: string; size: number; modified: number }>,
      readme: ReadmeContent,
      owner: string,
      repo: string
    ) => {
      if (!owner || !repo) {
        setError(
          "Invalid repository information. Owner and repo name are required."
        );
        setIsLoading(false);
        setEmbeddingError(false); // Reset embedding error state
        return;
      }

      // Skip if structure request is already in progress
      if (structureRequestInProgress) {
        console.log(
          "Wiki structure determination already in progress, skipping duplicate call"
        );
        return;
      }

      try {
        setStructureRequestInProgress(true);
        setStructureStartTime(Date.now());
        setTextReceivedCount(0);
        setElapsedTime(0);
        setLoadingMessage(
          messages.loading?.determiningStructure ||
            "Determining wiki structure..."
        );
        const structureStartTime = Date.now();
        let structureTokenCount = 0;

        // Get repository URL
        const repoUrl = getRepoUrl(effectiveRepoInfo);

        let fileTreeString: string;
        if (typeof fileTree === "string") {
          // Legacy format - already a string
          fileTreeString = fileTree;
        } else {
          const filteredFiles = fileTree;
          // // New structured format - convert to string, optionally filtering
          // // Filter out binary files and very large files for better analysis
          // const filteredFiles = fileTree.filter(item => {
          //   // Skip binary files
          //   if (item.is_binary) return false;
          //   // Skip files larger than 5MB
          //   if (item.size > 5 * 1024 * 1024) return false;
          //   return true;
          // });

          // Sort by modification time (most recent first) to prioritize active files
          const sortedFiles = [...filteredFiles].sort(
            (a, b) => b.modified - a.modified
          );

          // Convert to string format
          fileTreeString = sortedFiles.map((item) => item.path).join("\n");

          console.log(
            `Filtered file tree: ${filteredFiles.length}/${
              fileTree.length
            } files (excluded ${
              fileTree.length - filteredFiles.length
            } binary/large files)`
          );
        }

        const basicAnalysis = await analyzeRepository(
          fileTree,
          typeof readme === "string" ? readme : readme.project_purpose
        );

        // Decide if deep analysis is needed
        let deepAnalysis: RepoDeepAnalysis | null = null;
        if (basicAnalysis.complexityScore > 5) {
          // Only for moderately complex+ repos
          setLoadingMessage("Performing deep codebase analysis...");
          deepAnalysis = await performDeepCodebaseAnalysis(
            fileTreeString,
            readme,
            basicAnalysis,
            owner,
            repo
          );
        }

        const structurePrompt = await buildStructureGenerationPrompt(
          fileTreeString,
          readme,
          owner,
          repo,
          isComprehensiveView,
          basicAnalysis,
          deepAnalysis
        );

        // confirm the prompt if enabled
        let finalPrompt = structurePrompt;
        if (enablePromptEditing) {
          try {
            const model_to_use = `${selectedProviderState}/${
              isCustomSelectedModelState
                ? customSelectedModelState
                : selectedModelState
            }`;
            finalPrompt = await showPromptEditModal(
              structurePrompt,
              model_to_use,
              "Edit Wiki Structure Determination Prompt"
            );
          } catch (err) {
            // The user cancelled the edit -- gracefully abort structure determination
            setStructureRequestInProgress(false);
            setTextReceivedCount(0);
            setElapsedTime(0);
            // setLoadingMessage('User Cancelled Determining wiki structure... Refresh to retry');
            setLoadingMessage(
              '<span style="font-size:2.5em; margin-right:0.5em; vertical-align:middle;">❌</span><br/> User Cancelled Determining wiki structure... Refresh to retry'
            );
            // setIsLoading(false);
            // Optionally reset any custom structure state
            return;
          }
        }

        // Prepare request body
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const requestBody: ChatCompletionRequest = {
          repo_url: repoUrl,
          type: effectiveRepoInfo.type,
          messages: [
            {
              role: "user",
              content: finalPrompt,
            },
          ],
        };

        // Add tokens if available
        addTokensToRequestBody(
          requestBody,
          currentToken,
          effectiveRepoInfo.type,
          selectedProviderState,
          selectedModelState,
          isCustomSelectedModelState,
          customSelectedModelState,
          language,
          modelExcludedDirs,
          modelExcludedFiles,
          modelIncludedDirs,
          modelIncludedFiles
        );

        // Use WebSocket for communication
        let responseText = "";
        const requestStartTime = Date.now();
        let requestEndTime = requestStartTime;

        try {
          // Create WebSocket URL from the server base URL
          const serverBaseUrl =
            process.env.NEXT_PUBLIC_SERVER_BASE_URL || "http://localhost:8001";
          const wsBaseUrl = serverBaseUrl.replace(/^http/, "ws")
            ? serverBaseUrl.replace(/^https/, "wss")
            : serverBaseUrl.replace(/^http/, "ws");
          const wsUrl = `${wsBaseUrl}/ws/chat`;

          // Create a new WebSocket connection
          const ws = new WebSocket(wsUrl);

          // Create a promise that resolves when the WebSocket connection is complete
          await new Promise<void>((resolve, reject) => {
            // Set up event handlers
            ws.onopen = () => {
              console.log(
                "WebSocket connection established for wiki structure"
              );
              // Send the request as JSON
              ws.send(JSON.stringify(requestBody));
              resolve();
            };

            ws.onerror = (error) => {
              console.error("WebSocket error:", error);
              reject(new Error("WebSocket connection failed"));
            };

            // If the connection doesn't open within 5 seconds, fall back to HTTP
            const timeout = setTimeout(() => {
              reject(new Error("WebSocket connection timeout"));
            }, 5000);

            // Clear the timeout if the connection opens successfully
            ws.onopen = () => {
              clearTimeout(timeout);
              console.log(
                "WebSocket connection established for wiki structure"
              );
              // Send the request as JSON
              ws.send(JSON.stringify(requestBody));
              resolve();
            };
          });

          // Create a promise that resolves when the WebSocket response is complete
          await new Promise<void>((resolve, reject) => {
            // Handle incoming messages
            ws.onmessage = (event) => {
              try {
                const parsed = JSON.parse(event.data);

                if (parsed.type === "progress") {
                  // Handle embedding progress
                  setLoadingMessage(parsed.message);
                  if (
                    parsed.message.includes("embed") ||
                    parsed.message.includes("Embedding")
                  ) {
                    setEmbeddingProgress(parsed.message);
                  }
                  return; // Don't add to responseText for progress messages
                }
              } catch {
                // Not JSON, treat as regular content
              }
              responseText += event.data;
              structureTokenCount += event.data.length;
              setTextReceivedCount((prev) => prev + event.data.length);
            };

            // Handle WebSocket close
            ws.onclose = () => {
              console.log("WebSocket connection closed for wiki structure");
              requestEndTime = Date.now();
              resolve();
            };

            // Handle WebSocket errors
            ws.onerror = (error) => {
              console.error("WebSocket error during message reception:", error);
              requestEndTime = Date.now();
              reject(new Error("WebSocket error during message reception"));
            };
          });
        } catch (wsError) {
          console.error("WebSocket error, falling back to HTTP:", wsError);

          // Fall back to HTTP if WebSocket fails
          const response = await fetch(`/api/chat/stream`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(requestBody),
          });

          if (!response.ok) {
            requestEndTime = Date.now();
            throw new Error(
              `Error determining wiki structure: ${response.status}`
            );
          }

          // Process the response
          responseText = "";
          const reader = response.body?.getReader();
          const decoder = new TextDecoder();

          if (!reader) {
            throw new Error("Failed to get response reader");
          }

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            responseText += decoder.decode(value, { stream: true });
          }
          requestEndTime = Date.now();
        }

        if (
          responseText.includes(
            "Error preparing retriever: Environment variable OPENAI_API_KEY must be set"
          )
        ) {
          setEmbeddingError(true);
          throw new Error(
            "OPENAI_API_KEY environment variable is not set. Please configure your OpenAI API key."
          );
        }

        if (
          responseText.includes("Ollama model") &&
          responseText.includes("not found")
        ) {
          setEmbeddingError(true);
          throw new Error(
            "The specified Ollama embedding model was not found. Please ensure the model is installed locally or select a different embedding model in the configuration."
          );
        }

        // Add your debug line here:
        // console.log('Full AI response:', responseText);
        // Clean up markdown delimiters
        const cleanedResponseText = responseText
          .replace(/^```(?:xml)?\s*/i, "")
          .replace(/```\s*$/i, "");

        // Extract wiki structure from response
        const xmlMatch = cleanedResponseText.match(
          /<wiki_structure>[\s\S]*?<\/wiki_structure>/m
        );
        if (!xmlMatch) {
          throw new Error("No valid XML found in response");
        }

        // Proper XML escaping function
        function cleanXmlString(str: string) {
          return (
            str
              // Replace illegal/unescaped &
              .replace(
                /&(?!(amp|lt|gt|quot|apos|#\d+|#x[\da-fA-F]+);)/g,
                "&amp;"
              )
              // Remove forbidden ASCII control characters
              .replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "")
              // Replace string-escaped newlines/tabs (if present)
              .replace(/\\n/g, "\n")
              .replace(/\\t/g, "  ")
              .replace(/\\r/g, "")
          );
        }

        let xmlText = xmlMatch[0];
        xmlText = cleanXmlString(xmlText);
        // Try parsing with DOMParser
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xmlText, "text/xml");

        // Check for parsing errors
        const parseError = xmlDoc.querySelector("parsererror");
        if (parseError) {
          // Log the first few elements to see what was parsed
          const elements = xmlDoc.querySelectorAll("*");
          if (elements.length > 0) {
            console.log(
              "First 5 element names:",
              Array.from(elements)
                .slice(0, 5)
                .map((el) => el.nodeName)
                .join(", ")
            );
          }

          // We'll continue anyway since the XML might still be usable
        }

        // Extract wiki structure
        let title = "";
        let description = "";
        let pages: WikiPage[] = [];

        // Try using DOM parsing first
        const titleEl = xmlDoc.querySelector("title");
        const descriptionEl = xmlDoc.querySelector("description");
        const pagesEls = xmlDoc.querySelectorAll("page");

        title = titleEl ? titleEl.textContent || "" : "";
        description = descriptionEl ? descriptionEl.textContent || "" : "";

        // Parse pages using DOM
        pages = [];

        if (parseError && (!pagesEls || pagesEls.length === 0)) {
          console.warn("DOM parsing failed, trying regex fallback");
        }

        pagesEls.forEach((pageEl) => {
          const id = pageEl.getAttribute("id") || `page-${pages.length + 1}`;
          const titleEl = pageEl.querySelector("title");
          const importanceEl = pageEl.querySelector("importance");
          const pageTypeEl = pageEl.querySelector("page_type");
          const filePathEls = pageEl.querySelectorAll("file_path");
          const relatedEls = pageEl.querySelectorAll("related");

          const title = titleEl ? titleEl.textContent || "" : "";
          const importance = importanceEl
            ? importanceEl.textContent === "high"
              ? "high"
              : importanceEl.textContent === "medium"
              ? "medium"
              : "low"
            : "medium";

          // Extract page type
          const pageType = pageTypeEl
            ? (pageTypeEl.textContent as
                | "architecture"
                | "api"
                | "configuration"
                | "deployment"
                | "data_model"
                | "component"
                | "general")
            : undefined;

          const filePaths: string[] = [];
          filePathEls.forEach((el) => {
            if (el.textContent) filePaths.push(el.textContent);
          });

          const relatedPages: string[] = [];
          relatedEls.forEach((el) => {
            if (el.textContent) relatedPages.push(el.textContent);
          });

          pages.push({
            id,
            title,
            content: "", // Will be generated later
            filePaths,
            importance,
            pageType,
            relatedPages,
          });
        });

        setLoadingMessage(`Creating ${pages.length} pages`);

        // Extract sections if they exist in the XML
        const sections: WikiSection[] = [];
        const rootSections: string[] = [];

        // Try to parse sections if we're in comprehensive view
        if (isComprehensiveView) {
          const sectionsEls = xmlDoc.querySelectorAll("section");

          if (sectionsEls && sectionsEls.length > 0) {
            // Process sections
            sectionsEls.forEach((sectionEl) => {
              const id =
                sectionEl.getAttribute("id") ||
                `section-${sections.length + 1}`;
              const titleEl = sectionEl.querySelector("title");
              const pageRefEls = sectionEl.querySelectorAll("page_ref");
              const sectionRefEls = sectionEl.querySelectorAll("section_ref");

              const title = titleEl ? titleEl.textContent || "" : "";
              const pages: string[] = [];
              const subsections: string[] = [];

              pageRefEls.forEach((el) => {
                if (el.textContent) pages.push(el.textContent);
              });

              sectionRefEls.forEach((el) => {
                if (el.textContent) subsections.push(el.textContent);
              });

              sections.push({
                id,
                title,
                pages,
                subsections: subsections.length > 0 ? subsections : undefined,
              });

              // Check if this is a root section (not referenced by any other section)
              let isReferenced = false;
              sectionsEls.forEach((otherSection) => {
                const otherSectionRefs =
                  otherSection.querySelectorAll("section_ref");
                otherSectionRefs.forEach((ref) => {
                  if (ref.textContent === id) {
                    isReferenced = true;
                  }
                });
              });

              if (!isReferenced) {
                rootSections.push(id);
              }
            });
          }
        }

        // Create wiki structure
        const wikiStructure: WikiStructure = {
          id: "wiki",
          title,
          description,
          pages,
          sections,
          rootSections,
        };

        // console.log("Generated wikiStructure", wikiStructure);
        setWikiStructure(wikiStructure);
        addPromptLog({
          source: "WikiStructure",
          prompt: requestBody.messages
            .map((m) => `${m.role}: ${m.content}`)
            .join("\n\n"),
          response: responseText,
          timestamp: Date.now(),
          model: `${selectedProviderState}/${
            isCustomSelectedModelState
              ? customSelectedModelState
              : selectedModelState
          }`,
          timeTaken: (requestEndTime - requestStartTime) / 1000,
        });

        const timeTaken = Math.floor((Date.now() - structureStartTime) / 1000);
        setWikiAnalytics({
          model: selectedModelState || "default",
          provider: selectedProviderState || "default",
          tokensReceived: structureTokenCount,
          timeTaken,
        });

        setCurrentPageId(pages.length > 0 ? pages[0].id : undefined);
        // console.log("Found Pages:", pages.length, "Setting Cur page: ", pages.length > 0 ? pages[0].id : undefined)

        // Start generating content for all pages with controlled concurrency
        if (pages.length > 0) {
          // Mark all pages as in progress
          const initialInProgress = new Set(pages.map((p) => p.id));
          setPagesInProgress(initialInProgress);

          console.log(
            `Starting generation for ${pages.length} pages with controlled concurrency`
          );

          // Maximum concurrent requests
          const MAX_CONCURRENT = 1;

          // Create a queue of pages
          const queue = [...pages];
          let activeRequests = 0;

          // Function to process next items in queue
          const processQueue = () => {
            // Process as many items as we can up to our concurrency limit
            while (queue.length > 0 && activeRequests < MAX_CONCURRENT) {
              const page = queue.shift();
              if (page) {
                activeRequests++;
                console.log(
                  `Starting page ${page.title} (${activeRequests} active, ${queue.length} remaining)`
                );

                // Start generating content for this page
                generatePageContent(page, owner, repo).finally(() => {
                  // When done (success or error), decrement active count and process more
                  activeRequests--;
                  console.log(
                    `Finished page ${page.title} (${activeRequests} active, ${queue.length} remaining)`
                  );
                  // console.log("Generated Page:", page)

                  // Check if all work is done (queue empty and no active requests)
                  if (queue.length === 0 && activeRequests === 0) {
                    console.log("All page generation tasks completed.");
                    setIsLoading(false);
                    setLoadingMessage(undefined);
                  } else {
                    // Only process more if there are items remaining and we're under capacity
                    if (queue.length > 0 && activeRequests < MAX_CONCURRENT) {
                      processQueue();
                    }
                  }
                });
              }
            }

            // Additional check: If the queue started empty or becomes empty and no requests were started/active
            if (
              queue.length === 0 &&
              activeRequests === 0 &&
              pages.length > 0 &&
              pagesInProgress.size === 0
            ) {
              // This handles the case where the queue might finish before the finally blocks fully update activeRequests
              // or if the initial queue was processed very quickly
              console.log(
                "Queue empty and no active requests after loop, ensuring loading is false."
              );
              setIsLoading(false);
              setLoadingMessage(undefined);
            } else if (pages.length === 0) {
              // Handle case where there were no pages to begin with
              setIsLoading(false);
              setLoadingMessage(undefined);
            }
          };

          // Start processing the queue
          processQueue();
        } else {
          // Set loading to false if there were no pages found
          setIsLoading(false);
          setLoadingMessage(undefined);
        }
      } catch (error) {
        console.error("Error determining wiki structure:", error);
        setIsLoading(false);
        setError(
          error instanceof Error ? error.message : "An unknown error occurred"
        );
        setLoadingMessage(undefined);
      } finally {
        setStructureRequestInProgress(false);
        setStructureStartTime(null);
        setTextReceivedCount(0);
        setElapsedTime(0);
      }
    },
    [
      generatePageContent,
      currentToken,
      effectiveRepoInfo,
      pagesInProgress.size,
      structureRequestInProgress,
      selectedProviderState,
      selectedModelState,
      isCustomSelectedModelState,
      customSelectedModelState,
      modelExcludedDirs,
      modelExcludedFiles,
      language,
      messages.loading,
      isComprehensiveView,
    ]
  );

  // Fetch repository structure using GitHub or GitLab API
  const fetchRepositoryStructure = useCallback(async () => {
    // If a request is already in progress, don't start another one
    if (requestInProgress) {
      console.log(
        "Repository fetch already in progress, skipping duplicate call"
      );
      return;
    }

    // Reset previous state
    setWikiStructure(undefined);
    setCurrentPageId(undefined);
    setGeneratedPages({});
    setPagesInProgress(new Set());
    setError(null);
    setEmbeddingError(false); // Reset embedding error state

    try {
      // Set the request in progress flag
      setRequestInProgress(true);

      // Update loading state
      setIsLoading(true);
      setLoadingMessage(
        messages.loading?.fetchingStructure ||
          "Fetching repository structure..."
      );

      let fileTreeData = "";
      let readmeContent = "";

      if (effectiveRepoInfo.type === "local" && effectiveRepoInfo.localPath) {
        try {
          const params = new URLSearchParams({
            path: effectiveRepoInfo.localPath,
          });

          // Add file filter parameters if provided
          if (modelExcludedDirs) {
            params.append("excluded_dirs", modelExcludedDirs);
          }
          if (modelExcludedFiles) {
            params.append("excluded_files", modelExcludedFiles);
          }
          if (modelIncludedDirs) {
            params.append("included_dirs", modelIncludedDirs);
          }
          if (modelIncludedFiles) {
            params.append("included_files", modelIncludedFiles);
          }

          const response = await fetch(
            `/local_repo/structure?${params.toString()}`
          );

          if (!response.ok) {
            const errorData = await response.text();
            throw new Error(
              `Local repository API error (${response.status}): ${errorData}`
            );
          }

          const data = await response.json();
          // fileTreeData = data.file_tree;
          fileTreeData = Array.isArray(data.file_tree)
            ? data.file_tree
                .map((item: { path: string }) => item.path)
                .join("\n")
            : data.file_tree; // Fallback for backward compatibility
          readmeContent = data.readme;
          // For local repos, we can't determine the actual branch, so use 'main' as default
          setDefaultBranch("main");
        } catch (err) {
          throw err;
        }
      } else if (effectiveRepoInfo.type === "github") {
        // GitHub API approach
        // Try to get the tree data for common branch names
        let treeData = null;
        let apiErrorDetails = "";

        // Determine the GitHub API base URL based on the repository URL
        const getGithubApiUrl = (repoUrl: string | null): string => {
          if (!repoUrl) {
            return "https://api.github.com"; // Default to public GitHub
          }

          try {
            const url = new URL(repoUrl);
            const hostname = url.hostname;

            // If it's the public GitHub, use the standard API URL
            if (hostname === "github.com") {
              return "https://api.github.com";
            }

            // For GitHub Enterprise, use the enterprise API URL format
            // GitHub Enterprise API URL format: https://github.company.com/api/v3
            return `${url.protocol}//${hostname}/api/v3`;
          } catch {
            return "https://api.github.com"; // Fallback to public GitHub if URL parsing fails
          }
        };

        const githubApiBaseUrl = getGithubApiUrl(effectiveRepoInfo.repoUrl);
        // First, try to get the default branch from the repository info
        let defaultBranchLocal = null;
        try {
          const repoInfoResponse = await fetch(
            `${githubApiBaseUrl}/repos/${owner}/${repo}`,
            {
              headers: createGithubHeaders(currentToken),
            }
          );

          if (repoInfoResponse.ok) {
            const repoData = await repoInfoResponse.json();
            defaultBranchLocal = repoData.default_branch;
            console.log(`Found default branch: ${defaultBranchLocal}`);
            // Store the default branch in state
            setDefaultBranch(defaultBranchLocal || "main");
          }
        } catch (err) {
          console.warn(
            "Could not fetch repository info for default branch:",
            err
          );
        }

        // Create list of branches to try, prioritizing the actual default branch
        const branchesToTry = defaultBranchLocal
          ? [defaultBranchLocal, "main", "master"].filter(
              (branch, index, arr) => arr.indexOf(branch) === index
            )
          : ["main", "master"];

        for (const branch of branchesToTry) {
          const apiUrl = `${githubApiBaseUrl}/repos/${owner}/${repo}/git/trees/${branch}?recursive=1`;
          const headers = createGithubHeaders(currentToken);

          console.log(`Fetching repository structure from branch: ${branch}`);
          try {
            const response = await fetch(apiUrl, {
              headers,
            });

            if (response.ok) {
              treeData = await response.json();
              console.log("Successfully fetched repository structure");
              break;
            } else {
              const errorData = await response.text();
              apiErrorDetails = `Status: ${response.status}, Response: ${errorData}`;
              console.error(
                `Error fetching repository structure: ${apiErrorDetails}`
              );
            }
          } catch (err) {
            console.error(`Network error fetching branch ${branch}:`, err);
          }
        }

        if (!treeData || !treeData.tree) {
          if (apiErrorDetails) {
            throw new Error(
              `Could not fetch repository structure. API Error: ${apiErrorDetails}`
            );
          } else {
            throw new Error(
              "Could not fetch repository structure. Repository might not exist, be empty or private."
            );
          }
        }

        // Convert tree data to a string representation
        fileTreeData = treeData.tree
          .filter(
            (item: { type: string; path: string }) => item.type === "blob"
          )
          .map((item: { type: string; path: string }) => item.path)
          .join("\n");

        // Try to fetch README.md content
        try {
          const headers = createGithubHeaders(currentToken);

          const readmeResponse = await fetch(
            `${githubApiBaseUrl}/repos/${owner}/${repo}/readme`,
            {
              headers,
            }
          );

          if (readmeResponse.ok) {
            const readmeData = await readmeResponse.json();
            readmeContent = atob(readmeData.content);
          } else {
            console.warn(
              `Could not fetch README.md, status: ${readmeResponse.status}`
            );
          }
        } catch (err) {
          console.warn(
            "Could not fetch README.md, continuing with empty README",
            err
          );
        }
      } else if (effectiveRepoInfo.type === "gitlab") {
        // GitLab API approach
        const projectPath =
          extractUrlPath(effectiveRepoInfo.repoUrl ?? "") ?? `${owner}/${repo}`;
        const projectDomain = extractUrlDomain(
          effectiveRepoInfo.repoUrl ?? "https://gitlab.com"
        );
        const encodedProjectPath = encodeURIComponent(projectPath);

        const headers = createGitlabHeaders(currentToken);

        /* eslint-disable-next-line @typescript-eslint/no-explicit-any */
        const filesData: any[] = [];

        try {
          // Step 1: Get project info to determine default branch
          let projectInfoUrl: string;
          let defaultBranchLocal = "main"; // fallback
          try {
            const validatedUrl = new URL(projectDomain ?? ""); // Validate domain
            projectInfoUrl = `${validatedUrl.origin}/api/v4/projects/${encodedProjectPath}`;
          } catch (err) {
            throw new Error(`Invalid project domain URL: ${projectDomain}`);
          }
          const projectInfoRes = await fetch(projectInfoUrl, { headers });

          if (!projectInfoRes.ok) {
            const errorData = await projectInfoRes.text();
            throw new Error(
              `GitLab project info error: Status ${projectInfoRes.status}, Response: ${errorData}`
            );
          }

          const projectInfo = await projectInfoRes.json();
          defaultBranchLocal = projectInfo.default_branch || "main";
          console.log(`Found GitLab default branch: ${defaultBranchLocal}`);
          // Store the default branch in state
          setDefaultBranch(defaultBranchLocal);

          // Step 2: Paginate to fetch full file tree
          let page = 1;
          let morePages = true;

          while (morePages) {
            const apiUrl = `${projectInfoUrl}/repository/tree?recursive=true&per_page=100&page=${page}`;
            const response = await fetch(apiUrl, { headers });

            if (!response.ok) {
              const errorData = await response.text();
              throw new Error(
                `Error fetching GitLab repository structure (page ${page}): ${errorData}`
              );
            }

            const pageData = await response.json();
            filesData.push(...pageData);

            const nextPage = response.headers.get("x-next-page");
            morePages = !!nextPage;
            page = nextPage ? parseInt(nextPage, 10) : page + 1;
          }

          if (!Array.isArray(filesData) || filesData.length === 0) {
            throw new Error(
              "Could not fetch repository structure. Repository might be empty or inaccessible."
            );
          }

          // Step 3: Format file paths
          fileTreeData = filesData
            .filter(
              (item: { type: string; path: string }) => item.type === "blob"
            )
            .map((item: { type: string; path: string }) => item.path)
            .join("\n");

          // Step 4: Try to fetch README.md content
          const readmeUrl = `${projectInfoUrl}/repository/files/README.md/raw`;
          try {
            const readmeResponse = await fetch(readmeUrl, { headers });
            if (readmeResponse.ok) {
              readmeContent = await readmeResponse.text();
              console.log("Successfully fetched GitLab README.md");
            } else {
              console.warn(
                `Could not fetch GitLab README.md status: ${readmeResponse.status}`
              );
            }
          } catch (err) {
            console.warn(`Error fetching GitLab README.md:`, err);
          }
        } catch (err) {
          console.error("Error during GitLab repository tree retrieval:", err);
          throw err;
        }
      } else if (effectiveRepoInfo.type === "bitbucket") {
        // Bitbucket API approach
        const repoPath =
          extractUrlPath(effectiveRepoInfo.repoUrl ?? "") ?? `${owner}/${repo}`;
        const encodedRepoPath = encodeURIComponent(repoPath);

        // Try to get the file tree for common branch names
        let filesData = null;
        let apiErrorDetails = "";
        let defaultBranchLocal = "";
        const headers = createBitbucketHeaders(currentToken);

        // First get project info to determine default branch
        const projectInfoUrl = `https://api.bitbucket.org/2.0/repositories/${encodedRepoPath}`;
        try {
          const response = await fetch(projectInfoUrl, { headers });

          const responseText = await response.text();

          if (response.ok) {
            const projectData = JSON.parse(responseText);
            defaultBranchLocal = projectData.mainbranch.name;
            // Store the default branch in state
            setDefaultBranch(defaultBranchLocal);

            const apiUrl = `https://api.bitbucket.org/2.0/repositories/${encodedRepoPath}/src/${defaultBranchLocal}/?recursive=true&per_page=100`;
            try {
              const response = await fetch(apiUrl, {
                headers,
              });

              const structureResponseText = await response.text();

              if (response.ok) {
                filesData = JSON.parse(structureResponseText);
              } else {
                const errorData = structureResponseText;
                apiErrorDetails = `Status: ${response.status}, Response: ${errorData}`;
              }
            } catch (err) {
              console.error(
                `Network error fetching Bitbucket branch ${defaultBranchLocal}:`,
                err
              );
            }
          } else {
            const errorData = responseText;
            apiErrorDetails = `Status: ${response.status}, Response: ${errorData}`;
          }
        } catch (err) {
          console.error("Network error fetching Bitbucket project info:", err);
        }

        if (
          !filesData ||
          !Array.isArray(filesData.values) ||
          filesData.values.length === 0
        ) {
          if (apiErrorDetails) {
            throw new Error(
              `Could not fetch repository structure. Bitbucket API Error: ${apiErrorDetails}`
            );
          } else {
            throw new Error(
              "Could not fetch repository structure. Repository might not exist, be empty or private."
            );
          }
        }

        // Convert files data to a string representation
        fileTreeData = filesData.values
          .filter(
            (item: { type: string; path: string }) =>
              item.type === "commit_file"
          )
          .map((item: { type: string; path: string }) => item.path)
          .join("\n");

        // Try to fetch README.md content
        try {
          const headers = createBitbucketHeaders(currentToken);

          const readmeResponse = await fetch(
            `https://api.bitbucket.org/2.0/repositories/${encodedRepoPath}/src/${defaultBranchLocal}/README.md`,
            {
              headers,
            }
          );

          if (readmeResponse.ok) {
            readmeContent = await readmeResponse.text();
          } else {
            console.warn(
              `Could not fetch Bitbucket README.md, status: ${readmeResponse.status}`
            );
          }
        } catch (err) {
          console.warn(
            "Could not fetch Bitbucket README.md, continuing with empty README",
            err
          );
        }
      }

      let readmeAnalysis: ReadmeContent = readmeContent;

      const readmeTokens = Math.ceil(readmeContent.length / 4);
      const fileTreeTokens = Math.ceil(fileTreeData.length / 4);
      const totalTokens = readmeTokens + fileTreeTokens;
      console.log(
        `README tokens: ${readmeTokens}, File tree tokens: ${fileTreeTokens}, Total: ${totalTokens}`
      );
      console.log(
        "selectedModelState",
        selectedModelState,
        "selectedProviderState",
        selectedProviderState
      );
      const contextWindow = 5000; // Your Ollama model's context window

      if (totalTokens > contextWindow * 0.5 && readmeTokens > 5000) {
        // Large README - use multi-step approach
        console.log("README is large, analyzing separately...");
        setLoadingMessage("Analyzing README content...");

        readmeAnalysis = await analyzeReadmeContent(readmeContent, owner, repo);

        // setLoadingMessage(messages.loading?.determiningStructure || 'Determining wiki structure...');
        // await determineWikiStructure(fileTreeData, readmeAnalysis, owner, repo);
      } else {
        console.log("README is small, using direct approach...");
      }

      // Now determine the wiki structure
      await determineWikiStructure(fileTreeData, readmeAnalysis, owner, repo);
    } catch (error) {
      console.error("Error fetching repository structure:", error);
      setIsLoading(false);
      setError(
        error instanceof Error ? error.message : "An unknown error occurred"
      );
      setLoadingMessage(undefined);
    } finally {
      // Reset the request in progress flag
      setRequestInProgress(false);
    }
  }, [
    owner,
    repo,
    determineWikiStructure,
    currentToken,
    effectiveRepoInfo,
    requestInProgress,
    messages.loading,
  ]);

  // Function to export wiki content
  const exportWiki = useCallback(
    async (format: "markdown" | "json") => {
      if (!wikiStructure || Object.keys(generatedPages).length === 0) {
        setExportError("No wiki content to export");
        return;
      }

      try {
        setIsExporting(true);
        setExportError(null);
        setLoadingMessage(
          `${language === "ja" ? "Wikiを" : "Exporting wiki as "} ${format} ${
            language === "ja" ? "としてエクスポート中..." : "..."
          }`
        );

        // Prepare the pages for export
        const pagesToExport = wikiStructure.pages.map((page) => {
          // Use the generated content if available, otherwise use an empty string
          const content =
            generatedPages[page.id]?.content || "Content not generated";
          return {
            ...page,
            content,
          };
        });

        // Get repository URL
        const repoUrl = getRepoUrl(effectiveRepoInfo);

        // Make API call to export wiki
        const response = await fetch(`/export/wiki`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            repo_url: repoUrl,
            type: effectiveRepoInfo.type,
            pages: pagesToExport,
            format,
          }),
        });

        if (!response.ok) {
          const errorText = await response
            .text()
            .catch(() => "No error details available");
          throw new Error(
            `Error exporting wiki: ${response.status} - ${errorText}`
          );
        }

        // Get the filename from the Content-Disposition header if available
        const contentDisposition = response.headers.get("Content-Disposition");
        let filename = `${effectiveRepoInfo.repo}_wiki.${
          format === "markdown" ? "md" : "json"
        }`;

        if (contentDisposition) {
          const filenameMatch = contentDisposition.match(/filename=(.+)/);
          if (filenameMatch && filenameMatch[1]) {
            filename = filenameMatch[1].replace(/"/g, "");
          }
        }

        // Convert the response to a blob and download it
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } catch (err) {
        console.error("Error exporting wiki:", err);
        const errorMessage =
          err instanceof Error ? err.message : "Unknown error during export";
        setExportError(errorMessage);
      } finally {
        setIsExporting(false);
        setLoadingMessage(undefined);
      }
    },
    [wikiStructure, generatedPages, effectiveRepoInfo, language]
  );

  // No longer needed as we use the modal directly

  const confirmRefresh = useCallback(
    async (parameters: ModelSelectionParams) => {
      const { token } = parameters;

      setShowModelOptions(false);
      setLoadingMessage(
        messages.loading?.clearingCache || "Clearing server cache..."
      );
      setIsLoading(true); // Show loading indicator immediately

      try {
        const params = new URLSearchParams({
          owner: effectiveRepoInfo.owner,
          repo: effectiveRepoInfo.repo,
          repo_type: effectiveRepoInfo.type,
          language: language,
          provider: selectedProviderState,
          model: selectedModelState,
          is_custom_model: isCustomSelectedModelState.toString(),
          custom_model: customSelectedModelState,
          comprehensive: isComprehensiveView.toString(),
          authorization_code: authCode,
        });

        // Add file filters configuration
        if (modelExcludedDirs) {
          params.append("excluded_dirs", modelExcludedDirs);
        }
        if (modelExcludedFiles) {
          params.append("excluded_files", modelExcludedFiles);
        }

        if (authRequired && !authCode) {
          setIsLoading(false);
          console.error("Authorization code is required");
          setError("Authorization code is required");
          return;
        }

        const response = await fetch(`/api/wiki_cache?${params.toString()}`, {
          method: "DELETE",
          headers: {
            Accept: "application/json",
          },
        });

        if (response.ok) {
          console.log("Server-side wiki cache cleared successfully.");
          // Optionally, show a success message for cache clearing if desired
          // setLoadingMessage('Cache cleared. Refreshing wiki...');
        } else {
          const errorText = await response.text();
          console.warn(
            `Failed to clear server-side wiki cache (status: ${response.status}): ${errorText}. Proceeding with refresh anyway.`
          );
          // Optionally, inform the user about the cache clear failure but that refresh will still attempt
          // setError(\`Cache clear failed: ${errorText}. Trying to refresh...\`);
          if (response.status == 401) {
            setIsLoading(false);
            setLoadingMessage(undefined);
            setError("Failed to validate the authorization code");
            console.error("Failed to validate the authorization code");
            return;
          }
        }
      } catch (err) {
        console.warn("Error calling DELETE /api/wiki_cache:", err);
        setIsLoading(false);
        setEmbeddingError(false); // Reset embedding error state
        // Optionally, inform the user about the cache clear error
        // setError(\`Error clearing cache: ${err instanceof Error ? err.message : String(err)}. Trying to refresh...\`);
        throw err;
      }

      // Update token if provided
      if (token) {
        // Update current token state
        setCurrentToken(token);
        // Update the URL parameters to include the new token
        const currentUrl = new URL(window.location.href);
        currentUrl.searchParams.set("token", token);
        window.history.replaceState({}, "", currentUrl.toString());
      }

      // Proceed with the rest of the refresh logic
      console.log(
        "Refreshing wiki. Server cache will be overwritten upon new generation if not cleared."
      );

      // Clear the localStorage cache (if any remnants or if it was used before this change)
      const localStorageCacheKey = getCacheKey(
        effectiveRepoInfo.owner,
        effectiveRepoInfo.repo,
        effectiveRepoInfo.type,
        language,
        isComprehensiveView
      );
      localStorage.removeItem(localStorageCacheKey);

      // Reset cache loaded flag
      cacheLoadedSuccessfully.current = false;
      effectRan.current = false; // Allow the main data loading useEffect to run again

      // Reset all state
      setWikiStructure(undefined);
      setCurrentPageId(undefined);
      setGeneratedPages({});
      setPagesInProgress(new Set());
      setError(null);
      setEmbeddingError(false); // Reset embedding error state
      setIsLoading(true); // Set loading state for refresh
      setLoadingMessage(
        messages.loading?.initializing || "Initializing wiki generation..."
      );

      // Clear any in-progress requests for page content
      activeContentRequests.clear();
      // Reset flags related to request processing if they are component-wide
      setStructureRequestInProgress(false); // Assuming this flag should be reset
      setRequestInProgress(false); // Assuming this flag should be reset

      // Explicitly trigger the data loading process again by re-invoking what the main useEffect does.
      // This will first attempt to load from (now hopefully non-existent or soon-to-be-overwritten) server cache,
      // then proceed to fetchRepositoryStructure if needed.
      // To ensure fetchRepositoryStructure is called if cache is somehow still there or to force a full refresh:
      // One option is to directly call fetchRepositoryStructure() if force refresh means bypassing cache check.
      // For now, we rely on the standard loadData flow initiated by resetting effectRan and dependencies.
      // This will re-trigger the main data loading useEffect.
      // No direct call to fetchRepositoryStructure here, let the useEffect handle it based on effectRan.current = false.
    },
    [
      effectiveRepoInfo,
      language,
      messages.loading,
      activeContentRequests,
      selectedProviderState,
      selectedModelState,
      isCustomSelectedModelState,
      customSelectedModelState,
      modelExcludedDirs,
      modelExcludedFiles,
      isComprehensiveView,
      authCode,
      authRequired,
    ]
  );

  const handleRefreshWiki = () => {
    setShowWikiTypeInModal(true);
    setIsModelSelectionModalOpen(true);
    setRefreshPageIdQueued(null);
  };

  const handleModelSelectionApply = useCallback(
    async (params: ModelSelectionParams) => {
      setIsModelSelectionModalOpen(false);
      if (refreshPageIdQueued) {
        // Perform a single Page Refresh
        await confirmPrompt(refreshPageIdQueued, params);
      } else {
        // Otherwise, do full repo refresh
        await confirmRefresh(params);
      }
      setRefreshPageIdQueued(null);
    },
    [refreshPageIdQueued, confirmPrompt, confirmRefresh]
  );
  // Start wiki generation when component mounts
  useEffect(() => {
    if (effectRan.current === false) {
      effectRan.current = true; // Set to true immediately to prevent re-entry due to StrictMode

      const loadData = async () => {
        // Try loading from server-side cache first
        setLoadingMessage(
          messages.loading?.fetchingCache || "Checking for cached wiki..."
        );
        try {
          const params = new URLSearchParams({
            owner: effectiveRepoInfo.owner,
            repo: effectiveRepoInfo.repo,
            repo_type: effectiveRepoInfo.type,
            language: language,
            comprehensive: isComprehensiveView.toString(),
          });
          const response = await fetch(`/api/wiki_cache?${params.toString()}`);
          if (!response.ok)
            throw new Error(
              `Server error: ${response.status} - ${await response.text()}`
            );

          const cachedData = await response.json(); // Returns null if no cache
          if (
            cachedData &&
            cachedData.wiki_structure &&
            cachedData.generated_pages &&
            Object.keys(cachedData.generated_pages).length > 0
          ) {
            // Model/provider/repo updates...
            console.log("Using server-cached wiki data");
            if (cachedData.model) setSelectedModelState(cachedData.model);
            if (cachedData.provider)
              setSelectedProviderState(cachedData.provider);
            if (cachedData.repo) setEffectiveRepoInfo(cachedData.repo);
            else if (cachedData.repo_url && !effectiveRepoInfo.repoUrl)
              setEffectiveRepoInfo({
                ...effectiveRepoInfo,
                repoUrl: cachedData.repo_url,
              });

            // Ensure the cached structure has sections and rootSections
            const cachedStructure = {
              ...cachedData.wiki_structure,
              sections: cachedData.wiki_structure.sections || [],
              rootSections: cachedData.wiki_structure.rootSections || [],
              pages: Array.isArray(cachedData.wiki_structure.pages)
                ? cachedData.wiki_structure.pages
                : [],
            };

            // If sections or rootSections are missing, create intelligent ones based on page titles
            if (
              !cachedStructure.sections.length ||
              !cachedStructure.rootSections.length
            ) {
              const pages = cachedStructure.pages;
              const sections: WikiSection[] = [];
              const rootSections: string[] = [];

              // Group pages by common prefixes or categories
              const pageClusters = new Map<string, WikiPage[]>();

              // Define common categories that might appear in page titles
              const categories = [
                {
                  id: "overview",
                  title: "Overview",
                  keywords: ["overview", "introduction", "about"],
                },
                {
                  id: "architecture",
                  title: "Architecture",
                  keywords: ["architecture", "structure", "design", "system"],
                },
                {
                  id: "features",
                  title: "Core Features",
                  keywords: ["feature", "functionality", "core"],
                },
                {
                  id: "components",
                  title: "Components",
                  keywords: ["component", "module", "widget"],
                },
                {
                  id: "api",
                  title: "API",
                  keywords: ["api", "endpoint", "service", "server"],
                },
                {
                  id: "data",
                  title: "Data Flow",
                  keywords: ["data", "flow", "pipeline", "storage"],
                },
                {
                  id: "models",
                  title: "Models",
                  keywords: ["model", "ai", "ml", "integration"],
                },
                {
                  id: "ui",
                  title: "User Interface",
                  keywords: ["ui", "interface", "frontend", "page"],
                },
                {
                  id: "setup",
                  title: "Setup & Configuration",
                  keywords: ["setup", "config", "installation", "deploy"],
                },
              ];

              // Initialize clusters with empty arrays
              categories.forEach((category) => {
                pageClusters.set(category.id, []);
              });

              // Add an "Other" category for pages that don't match any category
              pageClusters.set("other", []);

              // Assign pages to categories based on title keywords
              pages.forEach((page: WikiPage) => {
                const title = page.title.toLowerCase();
                let assigned = false;

                // Try to find a matching category
                for (const category of categories) {
                  if (
                    category.keywords.some((keyword) => title.includes(keyword))
                  ) {
                    pageClusters.get(category.id)?.push(page);
                    assigned = true;
                    break;
                  }
                }

                // If no category matched, put in "Other"
                if (!assigned) {
                  pageClusters.get("other")?.push(page);
                }
              });

              // Create sections for non-empty categories
              for (const [
                categoryId,
                categoryPages,
              ] of pageClusters.entries()) {
                if (categoryPages.length > 0) {
                  const category = categories.find(
                    (c) => c.id === categoryId
                  ) || {
                    id: categoryId,
                    title:
                      categoryId === "other"
                        ? "Other"
                        : categoryId.charAt(0).toUpperCase() +
                          categoryId.slice(1),
                  };

                  const sectionId = `section-${categoryId}`;
                  sections.push({
                    id: sectionId,
                    title: category.title,
                    pages: categoryPages.map((p: WikiPage) => p.id),
                  });
                  rootSections.push(sectionId);

                  // Update page parentId
                  categoryPages.forEach((page: WikiPage) => {
                    page.parentId = sectionId;
                  });
                }
              }

              // If we still have no sections (unlikely), fall back to importance-based grouping
              if (sections.length === 0) {
                const highImportancePages = pages
                  .filter((p: WikiPage) => p.importance === "high")
                  .map((p: WikiPage) => p.id);
                const mediumImportancePages = pages
                  .filter((p: WikiPage) => p.importance === "medium")
                  .map((p: WikiPage) => p.id);
                const lowImportancePages = pages
                  .filter((p: WikiPage) => p.importance === "low")
                  .map((p: WikiPage) => p.id);

                if (highImportancePages.length > 0) {
                  sections.push({
                    id: "section-high",
                    title: "Core Components",
                    pages: highImportancePages,
                  });
                  rootSections.push("section-high");
                }

                if (mediumImportancePages.length > 0) {
                  sections.push({
                    id: "section-medium",
                    title: "Key Features",
                    pages: mediumImportancePages,
                  });
                  rootSections.push("section-medium");
                }

                if (lowImportancePages.length > 0) {
                  sections.push({
                    id: "section-low",
                    title: "Additional Information",
                    pages: lowImportancePages,
                  });
                  rootSections.push("section-low");
                }
              }

              cachedStructure.sections = sections;
              cachedStructure.rootSections = rootSections;
            }

            setWikiStructure(cachedStructure);
            setGeneratedPages(cachedData.generated_pages);
            setCurrentPageId(
              cachedStructure.pages.length > 0
                ? cachedStructure.pages[0].id
                : undefined
            );
            if (cachedData.wiki_analytics)
              setWikiAnalytics(cachedData.wiki_analytics);
            if (cachedData.page_analytics)
              setPageAnalytics(cachedData.page_analytics);

            setIsLoading(false);
            setEmbeddingError(false);
            setLoadingMessage(undefined);
            cacheLoadedSuccessfully.current = true;
            return; // Exit if cache is successfully loaded
          } else {
            console.log(
              "No valid wiki data in server cache or cache is empty."
            );
          }
        } catch (error) {
          console.error("Error loading from server cache:", error);
          // Proceed to fetch structure if cache loading fails
        }

        // If we reached here, either there was no cache, it was invalid, or an error occurred
        // Proceed to fetch repository structure
        fetchRepositoryStructure();
      };

      loadData();
    } else {
      console.log("Skipping duplicate repository fetch/cache check");
    }

    // Clean up function for this effect is not strictly necessary for loadData,
    // but keeping the main unmount cleanup in the other useEffect
  }, [
    effectiveRepoInfo,
    effectiveRepoInfo.owner,
    effectiveRepoInfo.repo,
    effectiveRepoInfo.type,
    language,
    fetchRepositoryStructure,
    messages.loading?.fetchingCache,
    isComprehensiveView,
  ]);

  // Save wiki to server-side cache when generation is complete
  useEffect(() => {
    const saveCache = async () => {
      if (
        !isLoading &&
        !error &&
        wikiStructure &&
        Object.keys(generatedPages).length > 0 &&
        Object.keys(generatedPages).length >= wikiStructure.pages.length &&
        !cacheLoadedSuccessfully.current
      ) {
        const allPagesHaveContent = wikiStructure.pages.every(
          (page) =>
            generatedPages[page.id] &&
            generatedPages[page.id].content &&
            generatedPages[page.id].content !== "Loading..."
        );

        if (allPagesHaveContent) {
          console.log(
            "Attempting to save wiki data to server cache via Next.js proxy"
          );

          try {
            // Make sure wikiStructure has sections and rootSections
            const structureToCache = {
              ...wikiStructure,
              sections: wikiStructure.sections || [],
              rootSections: wikiStructure.rootSections || [],
            };
            const dataToCache = {
              repo: effectiveRepoInfo,
              language: language,
              comprehensive: isComprehensiveView,
              wiki_structure: structureToCache,
              generated_pages: generatedPages,
              provider: selectedProviderState,
              model: selectedModelState,
              wiki_analytics: wikiAnalytics,
              page_analytics: pageAnalytics,
            };

            const response = await fetch(`/api/wiki_cache`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify(dataToCache),
            });

            if (response.ok) {
              console.log("Wiki data successfully saved to server cache");
            } else {
              console.error(
                "Error saving wiki data to server cache:",
                response.status,
                await response.text()
              );
            }
          } catch (error) {
            console.error("Error saving to server cache:", error);
          }
        }
      }
    };

    saveCache();
  }, [
    isLoading,
    error,
    wikiStructure,
    generatedPages,
    effectiveRepoInfo.owner,
    effectiveRepoInfo.repo,
    effectiveRepoInfo.type,
    effectiveRepoInfo.repoUrl,
    repoUrl,
    language,
    isComprehensiveView,
  ]);

  const handlePageSelect = (pageId: string) => {
    if (currentPageId != pageId) {
      setCurrentPageId(pageId);
    }
  };

  const formatElapsedTime = (totalSeconds: number): string => {
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, "0")}:${seconds
        .toString()
        .padStart(2, "0")}`;
    } else {
      return `${minutes}:${seconds.toString().padStart(2, "0")}`;
    }
  };

  const [isModelSelectionModalOpen, setIsModelSelectionModalOpen] =
    useState(false);

  return (
    <div className="h-screen paper-texture p-4 md:p-8 flex flex-col">
      <style>{wikiStyles}</style>

      <header className="max-w-[90%] xl:max-w-[1400px] mx-auto mb-8 h-fit w-full">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="flex items-center gap-4">
            <Link
              href="/"
              className="text-[var(--accent-primary)] hover:text-[var(--highlight)] flex items-center gap-1.5 transition-colors border-b border-[var(--border-color)] hover:border-[var(--accent-primary)] pb-0.5"
            >
              <FaHome /> {messages.repoPage?.home || "Home"}
            </Link>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-[90%] xl:max-w-[1400px] mx-auto overflow-y-auto">
        {isLoading ? (
          <div className="flex flex-col items-center justify-center p-8 bg-[var(--card-bg)] rounded-lg shadow-custom card-japanese">
            {!deepResearchEnabled && (
              <div className="relative mb-6">
                <div className="absolute -inset-4 bg-[var(--accent-primary)]/10 rounded-full blur-md animate-pulse"></div>
                <div className="relative flex items-center justify-center">
                  <div className="w-3 h-3 bg-[var(--accent-primary)]/70 rounded-full animate-pulse"></div>
                  <div className="w-3 h-3 bg-[var(--accent-primary)]/70 rounded-full animate-pulse delay-75 mx-2"></div>
                  <div className="w-3 h-3 bg-[var(--accent-primary)]/70 rounded-full animate-pulse delay-150"></div>
                </div>
              </div>
            )}
            <div className="text-[var(--foreground)] text-center font-serif">
              {/* {loadingMessage || messages.common?.loading || 'Loading...'} */}

              {!deepResearchEnabled && (
                <div
                  className="your-loading-message-class"
                  dangerouslySetInnerHTML={{
                    __html: loadingMessage
                      ? loadingMessage
                      : messages.common?.loading || "Loading...",
                  }}
                />
              )}
              {/* Embedding progress */}
              {embeddingProgress && !currentGeneratingPageId && (
                <div className="mt-4 p-3 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)]">
                  <div className="text-sm font-medium text-[var(--foreground)] mb-2">
                    Embedding Progress
                  </div>
                  <div className="text-xs text-[var(--muted)] space-y-1">
                    <div className="flex items-center space-x-2">
                      <div className="animate-pulse flex space-x-1">
                        <div className="h-2 w-2 bg-blue-500 rounded-full"></div>
                        <div className="h-2 w-2 bg-blue-500 rounded-full"></div>
                        <div className="h-2 w-2 bg-blue-500 rounded-full"></div>
                      </div>
                      <span>{embeddingProgress}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* Deep Research iteration progress */}
              {deepResearchEnabled && currentIteration > 0 && (
                <div className="mt-4 p-3 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)]">
                  <div className="text-sm font-medium text-[var(--foreground)] mb-2">
                    Deep Research Progress
                  </div>
                  <div className="text-xs text-[var(--muted)] space-y-2">
                    <div>
                      Iteration: {currentIteration} of {maxIterations}
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className="bg-purple-600 h-2 rounded-full transition-all duration-300"
                        style={{
                          width: `${(currentIteration / maxIterations) * 100}%`,
                        }}
                      ></div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="animate-pulse flex space-x-1">
                        <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
                        <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
                        <div className="h-2 w-2 bg-purple-600 rounded-full"></div>
                      </div>
                      <span>
                        {currentIteration === 1
                          ? "Planning research approach..."
                          : currentIteration < maxIterations
                          ? `Research iteration ${currentIteration} in progress...`
                          : "Finalizing comprehensive analysis..."}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {isExporting &&
                (messages.loading?.preparingDownload ||
                  " Please wait while we prepare your download...")}

              {/* Structure generation metrics */}
              {structureRequestInProgress && (
                <div className="mt-2 text-sm text-[var(--muted)]">
                  <div>
                    Text received: {textReceivedCount.toLocaleString()}{" "}
                    characters
                  </div>
                  <div>Time elapsed: {formatElapsedTime(elapsedTime)}</div>
                </div>
              )}

              {/* Page generation metrics */}
              {currentGeneratingPageId && (
                <div className="mt-2 text-sm text-[var(--muted)]">
                  <div>
                    Generating:{" "}
                    {
                      wikiStructure?.pages.find(
                        (p) => p.id === currentGeneratingPageId
                      )?.title
                    }
                  </div>
                  <div>
                    Text received: {pageTextReceivedCount.toLocaleString()}{" "}
                    characters
                  </div>
                  <div>Time elapsed: {formatElapsedTime(pageElapsedTime)}</div>
                </div>
              )}
            </div>
            {/* Progress bar for page generation */}
            {wikiStructure && (
              <div className="w-full max-w-md">
                <div style={{ marginBottom: "1em" }} className="mt-4 text-xs">
                  <p className="text-[var(--muted)] mb-2">Page Status:</p>
                  <ul className="text-[var(--foreground)] space-y-1">
                    {wikiStructure.pages.map((page) => {
                      let emoji = "";
                      let pageNameClass = "";
                      if (
                        deepResearchEnabled &&
                        page.id === currentGeneratingPageId
                      ) {
                        emoji = (
                          <span
                            className="inline-block w-3.5 h-3.5 rounded-full bg-blue-500 animate-pulse"
                            style={{ verticalAlign: "middle" }}
                          ></span>
                        );
                        pageNameClass = "font-medium animate-pulse truncate";
                      } else if (
                        generatedPages[page.id]?.content &&
                        generatedPages[page.id].content !== "Loading..."
                      ) {
                        emoji = "✅"; // Completed
                      } else if (page.id === currentGeneratingPageId) {
                        emoji = (
                          <span
                            className="inline-block w-3.5 h-3.5 rounded-full bg-blue-500 animate-pulse"
                            style={{ verticalAlign: "middle" }}
                          ></span>
                        );
                        pageNameClass = "font-medium animate-pulse truncate";
                      } else if (pagesInProgress.has(page.id)) {
                        emoji = "🕑"; // Pending/In Progress
                      } else {
                        emoji = "🕑"; // Treat as pending if status unclear
                      }
                      return (
                        <li
                          key={page.id}
                          className="flex justify-between items-center gap-2 py-0.5 text-[var(--foreground)]"
                          style={{ fontSize: "1.05em" }}
                        >
                          <span className={pageNameClass}>{page.title}</span>
                          <span className="ml-3 min-w-[1.5em] flex-shrink-0 flex justify-end items-center">
                            {emoji}
                          </span>
                        </li>
                      );
                    })}
                  </ul>
                </div>

                <div className="bg-[var(--background)]/50 rounded-full h-2 mb-3 overflow-hidden border border-[var(--border-color)]">
                  <div
                    className="bg-[var(--accent-primary)] h-2 rounded-full transition-all duration-300 ease-in-out"
                    style={{
                      width: `${Math.max(
                        5,
                        (100 *
                          (wikiStructure.pages.length - pagesInProgress.size)) /
                          wikiStructure.pages.length
                      )}%`,
                    }}
                  />
                </div>
                <p className="text-xs text-[var(--muted)] text-center">
                  {language === "ja"
                    ? `${wikiStructure.pages.length}ページ中${
                        wikiStructure.pages.length - pagesInProgress.size
                      }ページ完了`
                    : messages.repoPage?.pagesCompleted
                    ? messages.repoPage.pagesCompleted
                        .replace(
                          "{completed}",
                          (
                            wikiStructure.pages.length - pagesInProgress.size
                          ).toString()
                        )
                        .replace(
                          "{total}",
                          wikiStructure.pages.length.toString()
                        )
                    : `${
                        wikiStructure.pages.length - pagesInProgress.size
                      } of ${wikiStructure.pages.length} pages completed`}
                </p>
              </div>
            )}
          </div>
        ) : error ? (
          <div className="bg-[var(--highlight)]/5 border border-[var(--highlight)]/30 rounded-lg p-5 mb-4 shadow-sm">
            <div className="flex items-center text-[var(--highlight)] mb-3">
              <FaExclamationTriangle className="mr-2" />
              <span className="font-bold font-serif">
                {messages.repoPage?.errorTitle ||
                  messages.common?.error ||
                  "Error"}
              </span>
            </div>
            <p className="text-[var(--foreground)] text-sm mb-3">{error}</p>
            <p className="text-[var(--muted)] text-xs">
              {embeddingError
                ? messages.repoPage?.embeddingErrorDefault ||
                  "This error is related to the document embedding system used for analyzing your repository. Please verify your embedding model configuration, API keys, and try again. If the issue persists, consider switching to a different embedding provider in the model settings."
                : messages.repoPage?.errorMessageDefault ||
                  'Please check that your repository exists and is public. Valid formats are "owner/repo", "https://github.com/owner/repo", "https://gitlab.com/owner/repo", "https://bitbucket.org/owner/repo", or local folder paths like "C:\\path\\to\\folder" or "/path/to/folder".'}
            </p>
            <div className="mt-5">
              <Link
                href="/"
                className="btn-japanese px-5 py-2 inline-flex items-center gap-1.5"
              >
                <FaHome className="text-sm" />
                {messages.repoPage?.backToHome || "Back to Home"}
              </Link>
            </div>
          </div>
        ) : wikiStructure ? (
          <div className="h-full overflow-y-auto flex flex-col lg:flex-row gap-4 w-full overflow-hidden bg-[var(--card-bg)] rounded-lg shadow-custom card-japanese">
            {/* Wiki Navigation */}
            <div className="h-full w-full lg:w-[280px] xl:w-[320px] flex-shrink-0 bg-[var(--background)]/50 rounded-lg rounded-r-none p-5 border-b lg:border-b-0 lg:border-r border-[var(--border-color)] overflow-y-auto">
              <h3 className="text-lg font-bold text-[var(--foreground)] mb-3 font-serif">
                {wikiStructure.title}
              </h3>
              <p className="text-[var(--muted)] text-sm mb-5 leading-relaxed">
                {wikiStructure.description}
              </p>

              {/* Display repository info */}
              <div className="text-xs text-[var(--muted)] mb-5 flex items-center">
                {effectiveRepoInfo.type === "local" ? (
                  <div className="flex items-center">
                    <FaFolder className="mr-2" />
                    <span className="break-all">
                      {effectiveRepoInfo.localPath}
                    </span>
                  </div>
                ) : (
                  <>
                    {effectiveRepoInfo.type === "github" ? (
                      <FaGithub className="mr-2" />
                    ) : effectiveRepoInfo.type === "gitlab" ? (
                      <FaGitlab className="mr-2" />
                    ) : (
                      <FaBitbucket className="mr-2" />
                    )}
                    <a
                      href={effectiveRepoInfo.repoUrl ?? ""}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:text-[var(--accent-primary)] transition-colors border-b border-[var(--border-color)] hover:border-[var(--accent-primary)]"
                    >
                      {effectiveRepoInfo.owner}/{effectiveRepoInfo.repo}
                    </a>
                  </>
                )}
              </div>

              {/* Wiki Type Indicator */}
              <div className="mb-3 flex items-center text-xs text-[var(--muted)]">
                <span className="mr-2">Wiki Type:</span>
                <span
                  className={`px-2 py-0.5 rounded-full ${
                    isComprehensiveView
                      ? "bg-[var(--accent-primary)]/10 text-[var(--accent-primary)] border border-[var(--accent-primary)]/30"
                      : "bg-[var(--background)] text-[var(--foreground)] border border-[var(--border-color)]"
                  }`}
                >
                  {isComprehensiveView
                    ? messages.form?.comprehensive || "Comprehensive"
                    : messages.form?.concise || "Concise"}
                </span>
              </div>

              {/* Refresh Wiki button */}
              <div className="mb-5">
                <button
                  onClick={() => handleRefreshWiki()}
                  disabled={isLoading}
                  className="flex items-center w-full text-xs px-3 py-2 bg-[var(--background)] text-[var(--foreground)] rounded-md hover:bg-[var(--background)]/80 disabled:opacity-50 disabled:cursor-not-allowed border border-[var(--border-color)] transition-colors hover:cursor-pointer"
                >
                  <FaSync
                    className={`mr-2 ${isLoading ? "animate-spin" : ""}`}
                  />
                  {messages.repoPage?.refreshWiki || "Refresh Wiki"}
                </button>
              </div>

              {/* Prompt Editing Control */}
              <button
                className="px-3 py-1 rounded border text-sm bg-[var(--background)] border-[var(--border-color)] hover:bg-[var(--accent-primary)]/10"
                onClick={() => setEnablePromptEditing((v) => !v)}
              >
                {enablePromptEditing ? "Disable" : "Enable"} Prompt Editing
              </button>

              {/* Analytics Section */}
              <div className="mb-5">
                <h4 className="text-sm font-semibold text-[var(--foreground)] mb-3 font-serif">
                  Generation Analytics
                </h4>

                {wikiAnalytics || Object.keys(pageAnalytics).length > 0 ? (
                  <>
                    {/* Wiki Analysis Stats */}
                    {wikiAnalytics && (
                      <div className="mb-3 p-3 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)]">
                        <div className="text-xs font-medium text-[var(--foreground)] mb-2">
                          Wiki Analysis
                        </div>
                        <div className="text-xs text-[var(--muted)] space-y-1">
                          <div>
                            Model: {wikiAnalytics.provider}/
                            {wikiAnalytics.model}
                          </div>
                          <div>
                            Tokens:{" "}
                            {wikiAnalytics.tokensReceived.toLocaleString()}
                          </div>
                          <div>
                            Time: {formatElapsedTime(wikiAnalytics.timeTaken)}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Page Generation Stats */}
                    {Object.keys(pageAnalytics).length > 0 && (
                      <div className="p-3 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)]">
                        <div className="text-xs font-medium text-[var(--foreground)] mb-2">
                          Pages Generated ({Object.keys(pageAnalytics).length})
                        </div>
                        <div className="max-h-32 overflow-y-auto space-y-2">
                          {Object.entries(pageAnalytics).map(
                            ([pageId, analytics]) => {
                              const page = wikiStructure?.pages.find(
                                (p) => p.id === pageId
                              );
                              return (
                                <div
                                  key={pageId}
                                  className="text-xs text-[var(--muted)]"
                                >
                                  <div className="font-medium truncate">
                                    {page?.title || pageId}
                                  </div>
                                  <div className="ml-2 space-y-0.5">
                                    <div>
                                      Model: {analytics.provider}/
                                      {analytics.model}
                                    </div>
                                    <div>
                                      Tokens:{" "}
                                      {analytics.tokensReceived.toLocaleString()}
                                    </div>
                                    <div>
                                      Time:{" "}
                                      {formatElapsedTime(analytics.timeTaken)}
                                    </div>
                                  </div>
                                </div>
                              );
                            }
                          )}
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <div className="p-3 bg-[var(--background)]/50 rounded-md border border-[var(--border-color)]">
                    <div className="text-xs text-[var(--muted)] text-center">
                      Analytics not available for this wiki
                    </div>
                  </div>
                )}
              </div>

              {/* Export buttons */}
              {Object.keys(generatedPages).length > 0 && (
                <div className="mb-5">
                  <h4 className="text-sm font-semibold text-[var(--foreground)] mb-3 font-serif">
                    {messages.repoPage?.exportWiki || "Export Wiki"}
                  </h4>
                  <div className="flex flex-col gap-2">
                    <button
                      onClick={() => exportWiki("markdown")}
                      disabled={isExporting}
                      className="btn-japanese flex items-center text-xs px-3 py-2 rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      <FaDownload className="mr-2" />
                      {messages.repoPage?.exportAsMarkdown ||
                        "Export as Markdown"}
                    </button>
                    <button
                      onClick={() => exportWiki("json")}
                      disabled={isExporting}
                      className="flex items-center text-xs px-3 py-2 bg-[var(--background)] text-[var(--foreground)] rounded-md hover:bg-[var(--background)]/80 disabled:opacity-50 disabled:cursor-not-allowed border border-[var(--border-color)] transition-colors"
                    >
                      <FaFileExport className="mr-2" />
                      {messages.repoPage?.exportAsJson || "Export as JSON"}
                    </button>
                  </div>
                  {exportError && (
                    <div className="mt-2 text-xs text-[var(--highlight)]">
                      {exportError}
                    </div>
                  )}
                </div>
              )}

              <h4 className="text-md font-semibold text-[var(--foreground)] mb-3 font-serif">
                {messages.repoPage?.pages || "Pages"}
              </h4>
              <WikiTreeView
                wikiStructure={wikiStructure}
                currentPageId={currentPageId}
                onPageSelect={handlePageSelect}
                messages={messages.repoPage}
                onPageRefresh={refreshPage}
                pagesInProgress={pagesInProgress}
              />
            </div>

            {/* Wiki Content */}
            <div
              id="wiki-content"
              className="w-full flex-grow p-6 lg:p-8 overflow-y-auto"
            >
              {currentPageId && generatedPages[currentPageId] ? (
                <div className="max-w-[900px] xl:max-w-[1000px] mx-auto">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold text-[var(--foreground)] mb-4 break-words font-serif">
                      {generatedPages[currentPageId].title}
                    </h3>

                    <div className="flex items-center gap-2 mb-4">
                      {/* Deep Research Toggle */}
                      <div className="group relative">
                        <label className="flex items-center cursor-pointer">
                          <span className="text-xs text-gray-600 dark:text-gray-400 mr-2">
                            Deep Research
                          </span>
                          <div className="relative">
                            <input
                              type="checkbox"
                              checked={deepResearchEnabled}
                              onChange={() =>
                                setDeepResearchEnabled(!deepResearchEnabled)
                              }
                              className="sr-only"
                            />
                            <div
                              className={`w-10 h-5 rounded-full transition-colors ${
                                deepResearchEnabled
                                  ? "bg-purple-600"
                                  : "bg-gray-300 dark:bg-gray-600"
                              }`}
                            ></div>
                            <div
                              className={`absolute left-0.5 top-0.5 w-4 h-4 rounded-full bg-white transition-transform transform ${
                                deepResearchEnabled ? "translate-x-5" : ""
                              }`}
                            ></div>
                          </div>
                        </label>
                        <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block bg-gray-800 text-white text-xs rounded p-2 w-72 z-10">
                          <p className="mb-1">
                            Deep Research conducts multi-iteration analysis for
                            comprehensive page content
                          </p>
                        </div>
                      </div>

                      {/* Status indicator when Deep Research is active */}
                      {deepResearchEnabled && (
                        <div className="text-xs text-purple-600 dark:text-purple-400">
                          <div className="flex items-center gap-2 mb-1">
                            {/* Animated dot when in progress */}
                            {currentIteration > 0 &&
                              !researchComplete &&
                              pagesInProgress.has(currentPageId) && (
                                <div className="flex items-center gap-1">
                                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                                  <span className="animate-pulse">
                                    Researching
                                  </span>
                                </div>
                              )}
                            <span>Multi-turn research</span>
                            {currentIteration > 0 && (
                              <>
                                <span>•</span>
                                <span className="font-medium">
                                  {currentIteration}/5
                                </span>
                              </>
                            )}
                            {researchComplete && (
                              <>
                                <span>•</span>
                                <span className="text-green-600 dark:text-green-400">
                                  ✓ Complete
                                </span>
                              </>
                            )}
                          </div>

                          {/* Progress bar */}
                          {currentIteration > 0 && (
                            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5">
                              <div
                                className={`h-1.5 rounded-full transition-all duration-300 ${
                                  researchComplete
                                    ? "bg-green-500"
                                    : "bg-purple-500"
                                }`}
                                style={{
                                  width: researchComplete
                                    ? "100%"
                                    : `${(currentIteration / 5) * 100}%`,
                                }}
                              />
                            </div>
                          )}
                        </div>
                      )}

                      {/* Refresh Page Button */}
                      <button
                        onClick={() => refreshPage(currentPageId)}
                        disabled={pagesInProgress.has(currentPageId)}
                        className="flex items-center px-3 py-2 bg-[var(--accent-primary)] text-white rounded-md hover:bg-[var(--accent-primary)]/90 disabled:opacity-50 disabled:cursor-not-allowed border border-[var(--border-color)] transition-colors hover:cursor-pointer"
                        title="Refresh this page"
                      >
                        <FaSync
                          className={`mr-2 ${
                            pagesInProgress.has(currentPageId)
                              ? "animate-spin"
                              : ""
                          }`}
                        />
                        Refresh Page
                      </button>
                    </div>
                  </div>

                  <div className="prose prose-sm md:prose-base lg:prose-lg max-w-none">
                    {/* <Markdown content={generatedPages[currentPageId].content} /> */}
                    <IterationTabs page={generatedPages[currentPageId]} />
                  </div>

                  {generatedPages[currentPageId].relatedPages.length > 0 && (
                    <div className="mt-8 pt-4 border-t border-[var(--border-color)]">
                      <h4 className="text-sm font-semibold text-[var(--muted)] mb-3">
                        {messages.repoPage?.relatedPages || "Related Pages:"}
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {generatedPages[currentPageId].relatedPages.map(
                          (relatedId) => {
                            const relatedPage = wikiStructure.pages.find(
                              (p) => p.id === relatedId
                            );
                            return relatedPage ? (
                              <button
                                key={relatedId}
                                className="bg-[var(--accent-primary)]/10 hover:bg-[var(--accent-primary)]/20 text-xs text-[var(--accent-primary)] px-3 py-1.5 rounded-md transition-colors truncate max-w-full border border-[var(--accent-primary)]/20"
                                onClick={() => handlePageSelect(relatedId)}
                              >
                                {relatedPage.title}
                              </button>
                            ) : null;
                          }
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center p-8 text-[var(--muted)] h-full">
                  <div className="relative mb-4">
                    <div className="absolute -inset-2 bg-[var(--accent-primary)]/5 rounded-full blur-md"></div>
                    <FaBookOpen className="text-4xl relative z-10" />
                  </div>
                  <p className="font-serif">
                    {messages.repoPage?.selectPagePrompt ||
                      "Select a page from the navigation to view its content"}
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : null}
      </main>

      <footer className="max-w-[90%] xl:max-w-[1400px] mx-auto mt-8 flex flex-col gap-4 w-full">
        <div className="flex justify-between items-center gap-4 text-center text-[var(--muted)] text-sm h-fit w-full bg-[var(--card-bg)] rounded-lg p-3 shadow-sm border border-[var(--border-color)]">
          <p className="flex-1 font-serif">
            {messages.footer?.copyright ||
              "DeepWiki - Generate Wiki from GitHub/Gitlab/Bitbucket repositories"}
          </p>
          <ThemeToggle />
        </div>
      </footer>

      {/* Floating Chat Button */}
      {!isLoading && wikiStructure && (
        <button
          onClick={() => setIsAskModalOpen(true)}
          className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-[var(--accent-primary)] text-white shadow-lg flex items-center justify-center hover:bg-[var(--accent-primary)]/90 transition-all z-50"
          aria-label={messages.ask?.title || "Ask about this repository"}
        >
          <FaComments className="text-xl" />
        </button>
      )}

      {/* Ask Modal - Always render but conditionally show/hide */}
      <div
        className={`fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4 transition-opacity duration-300 ${
          isAskModalOpen ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
        <div className="bg-[var(--card-bg)] rounded-lg shadow-xl w-full max-w-3xl max-h-[80vh] flex flex-col">
          <div className="flex items-center justify-end p-3 absolute top-0 right-0 z-10">
            <button
              onClick={() => {
                // Just close the modal without clearing the conversation
                setIsAskModalOpen(false);
              }}
              className="text-[var(--muted)] hover:text-[var(--foreground)] transition-colors bg-[var(--card-bg)]/80 rounded-full p-2"
              aria-label="Close"
            >
              <FaTimes className="text-xl" />
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-4">
            <Ask
              repoInfo={effectiveRepoInfo}
              onRetrievedDocs={(newSet) =>
                setRetrievedDocs((prev) => [...prev, newSet])
              }
              provider={selectedProviderState}
              model={selectedModelState}
              isCustomModel={isCustomSelectedModelState}
              customModel={customSelectedModelState}
              language={language}
              onRef={(ref) => (askComponentRef.current = ref)}
            />
          </div>
        </div>
      </div>

      <ModelSelectionModal
        isOpen={isModelSelectionModalOpen}
        onClose={() => setIsModelSelectionModalOpen(false)}
        provider={selectedProviderState}
        setProvider={setSelectedProviderState}
        model={selectedModelState}
        setModel={setSelectedModelState}
        isCustomModel={isCustomSelectedModelState}
        setIsCustomModel={setIsCustomSelectedModelState}
        customModel={customSelectedModelState}
        setCustomModel={setCustomSelectedModelState}
        isComprehensiveView={isComprehensiveView}
        setIsComprehensiveView={setIsComprehensiveView}
        showFileFilters={true}
        excludedDirs={modelExcludedDirs}
        setExcludedDirs={setModelExcludedDirs}
        excludedFiles={modelExcludedFiles}
        setExcludedFiles={setModelExcludedFiles}
        includedDirs={modelIncludedDirs}
        setIncludedDirs={setModelIncludedDirs}
        includedFiles={modelIncludedFiles}
        setIncludedFiles={setModelIncludedFiles}
        onApply={handleModelSelectionApply}
        // onApply={(params) => {
        //   console.log('Model selection applied:', params);
        // }}
        showWikiType={showWikiTypeInModal}
        showTokenInput={effectiveRepoInfo.type !== "local" && !currentToken} // Show token input if not local and no current token
        repositoryType={
          effectiveRepoInfo.type as "github" | "gitlab" | "bitbucket"
        }
        authRequired={authRequired}
        authCode={authCode}
        setAuthCode={setAuthCode}
        isAuthLoading={isAuthLoading}
      />
      <PromptEditorModal
        isOpen={showPromptModal}
        prompt={promptToEdit}
        model={modelToShow}
        title={promptModalTitle}
        // onApply={handleApplyPromptEdit}
        onApply={(editedPrompt: string) => {
          setShowPromptModal(false);
          if (onPromptConfirm) onPromptConfirm(editedPrompt);
          setOnPromptConfirm(null);
        }}
        onCancel={() => {
          setShowPromptModal(false);
          if (onPromptCancel) onPromptCancel();
          setOnPromptCancel(null);
          setOnPromptConfirm(null);
        }}
      />
      <PromptLogFloatingPanel />

      {/*
        <RetrievedDocsFloatingPanel retrievedSets={retrievedDocs} />
      */}
    </div>
  );
}
