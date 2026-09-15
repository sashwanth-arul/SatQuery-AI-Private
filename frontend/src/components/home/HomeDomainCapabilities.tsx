import Link from "next/link";

interface DomainCapability {
  domain: string;
  title: string;
  description: string;
  pipeline: string;
  exampleQuery: string;
}

const DOMAINS: DomainCapability[] = [
  {
    domain: "agriculture",
    title: "Agriculture Monitoring",
    description: "Assess crop health, agricultural parcels, and vegetation condition without manual GIS thresholding.",
    pipeline: "NDVI / VARI multispectral indices with configurable vegetation heuristics",
    exampleQuery: "Show agricultural areas in this region",
  },
  {
    domain: "disaster",
    title: "Disaster Management",
    description: "Map flood inundation extents, detect receding water, and quantify post-flood changes across acquisitions.",
    pipeline: "Optical NDWI & SAR specular drop with bi-temporal flood delta calculation",
    exampleQuery: "Show flooded areas",
  },
  {
    domain: "urban",
    title: "Urban Planning",
    description: "Track urban growth, built-up surface expansion, and impervious surfaces over temporal intervals.",
    pipeline: "NDBI / Dynamic World built surface tracking with surface area metrics",
    exampleQuery: "Show built-up area changes",
  },
  {
    domain: "forestry",
    title: "Forest Monitoring",
    description: "Measure dense forest canopy coverage, track woodland parcels, and evaluate forest density shifts.",
    pipeline: "High-density NDVI canopy segmentation with vector boundary polygonization",
    exampleQuery: "Show forested areas",
  },
  {
    domain: "water",
    title: "Water-Resource Assessment",
    description: "Extract surface water bodies, reservoirs, lakes, and river channels with accurate square-meter calculations.",
    pipeline: "Normalized Difference Water Index (NDWI) with SWIR validation",
    exampleQuery: "Show water bodies in this scene",
  },
  {
    domain: "infrastructure",
    title: "Infrastructure Mapping",
    description: "Detect physical structural footprints, count buildings, and match bipartite footprints between temporal scenes.",
    pipeline: "Deterministic contour geometry extraction & bipartite IoU footprint matching",
    exampleQuery: "Show buildings and infrastructure",
  },
  {
    domain: "environmental",
    title: "Environmental Analysis",
    description: "Classify multi-class land cover into water, forest, agriculture, and built-up areas with comprehensive distributions.",
    pipeline: "Joint multispectral band classification and distribution percentage audit",
    exampleQuery: "Describe the land cover",
  },
];

export function HomeDomainCapabilities() {
  return (
    <section
      className="home-about__section"
      id="capabilities"
      aria-labelledby="capabilities-heading"
      style={{ borderTop: "1px solid var(--border)", paddingTop: "3rem", paddingBottom: "3rem" }}
    >
      <div className="home-about__container">
        <div className="home-about__section-header mb-8">
          <p className="home-about__eyebrow">Natural-Language Intent Routing</p>
          <h2 id="capabilities-heading" className="home-about__title home-about__title--section">
            What can SatQuery analyze?
          </h2>
          <p className="home-about__lead" style={{ maxWidth: "48rem" }}>
            You never need to manually choose an algorithm or select a domain dropdown. Simply ask in plain English — the agentic query planner identifies the intent and invokes the verified specialist tool.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {DOMAINS.map((item) => (
            <div
              key={item.domain}
              className="rounded-xl border border-[var(--border)] bg-[var(--surface-sunken)] p-4 flex flex-col justify-between hover:border-[var(--accent)] transition-colors"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-base font-semibold m-0 text-[var(--text-primary)]">
                    {item.title}
                  </h3>
                  <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-[var(--surface)] border border-[var(--border)] text-[var(--accent)]">
                    {item.domain}
                  </span>
                </div>
                <p className="text-xs text-[var(--text-secondary)] leading-relaxed m-0 mb-3">
                  {item.description}
                </p>
                <div className="text-[11px] text-[var(--text-muted)] font-mono mb-3">
                  <span className="font-semibold text-[var(--text-secondary)]">Engine: </span>
                  {item.pipeline}
                </div>
              </div>

              <div className="pt-2 border-t border-[var(--border)] mt-auto">
                <div className="text-[11px] text-[var(--text-muted)] mb-1">Quick query:</div>
                <Link
                  href={`/workstation?query=${encodeURIComponent(item.exampleQuery)}`}
                  className="inline-flex items-center justify-between w-full px-2.5 py-1.5 rounded text-xs bg-[var(--surface)] border border-[var(--border)] hover:bg-[var(--accent)]/10 hover:border-[var(--accent)] text-[var(--text-primary)] transition-colors group"
                >
                  <span className="truncate italic text-[var(--text-secondary)] group-hover:text-[var(--text-primary)]">
                    &ldquo;{item.exampleQuery}&rdquo;
                  </span>
                  <span className="text-[var(--accent)] font-semibold ml-2">Run →</span>
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
