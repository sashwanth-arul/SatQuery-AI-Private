import Image from "next/image";
import Link from "next/link";
import { HomeNav } from "@/components/home/HomeNav";

export function MarketingNav() {
  return (
    <>
      <div className="marketing-nav__brand">
        <Link href="/" className="marketing-nav__logo" data-testid="site-nav-brand">
          <Image
            src="/LOGO.png"
            alt="SatQuery"
            width={1340}
            height={343}
            className="marketing-nav__logo-image"
            priority
          />
        </Link>
        <a
          href="https://www.isro.gov.in/"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Visit ISRO official website"
          className="marketing-nav__isro"
        >
          <Image
            src="/isro-logo.png"
            alt="ISRO"
            width={3000}
            height={2000}
            className="marketing-nav__isro-image"
            priority
          />
        </a>
      </div>
      <HomeNav />
    </>
  );
}
