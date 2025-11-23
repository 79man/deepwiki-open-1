// Wiki Interfaces

export interface WikiPageIteration {
  iteration: number;
  content: string;
  timestamp: number;
  model?: string;
  provider?: string;
};

export interface WikiPage {
  /**
   * Represents a single wiki page with metadata
   * @property id - Unique identifier (e.g., "page-1")
   * @property title - Display title for the page
   * @property importance - Priority level: "high" | "medium" | "low"
   * @property pageType - Content type for tailored prompts:
   *   - "architecture": System design pages
   *   - "api": API documentation
   *   - "configuration": Config file docs
   *   - "deployment": Deployment guides
   *   - "data_model": Database schemas
   *   - "component": UI/module docs
   *   - "general": Overview pages
   * @property filePaths - Source files used to generate content
   */
  id: string;
  title: string;
  content: string; // Used for non-DeepResearch pages
  iterations?: Array<WikiPageIteration>; // Used only for DeepResearch Pages
  filePaths: string[];
  importance: "high" | "medium" | "low";
  pageType?:
    | "architecture"
    | "api"
    | "configuration"
    | "deployment"
    | "data_model"
    | "component"
    | "general";
  relatedPages: string[];
  parentId?: string;
  isSection?: boolean;
  children?: string[];
}
