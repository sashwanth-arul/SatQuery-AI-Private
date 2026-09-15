import { Playfair_Display } from "next/font/google";
import { HomeAboutSection } from "@/components/home/HomeAboutSection";
import { HomeDomainCapabilities } from "@/components/home/HomeDomainCapabilities";
import { HomeFooter } from "@/components/home/HomeFooter";
import { HomeTransitionBand } from "@/components/home/HomeTransitionBand";
import { MarketingNav } from "@/components/MarketingNav";
import { ResponsiveHeroBanner } from "@/components/ui/responsive-hero-banner";

const playfairDisplay = Playfair_Display({
  subsets: ["latin"],
  weight: ["400"],
  variable: "--font-home-display",
  display: "swap",
});

export default function LandingPage() {
  return (
    <div className={`home-route ${playfairDisplay.variable}`}>
      <MarketingNav />
      <ResponsiveHeroBanner primaryButtonHref="/workstation" />
      <HomeTransitionBand />
      <HomeAboutSection />
      <HomeDomainCapabilities />
      <HomeFooter />
    </div>
  );
}
