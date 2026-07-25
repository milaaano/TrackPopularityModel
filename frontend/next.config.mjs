/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static-friendly one-pager. Uploads go straight to FastAPI (PLAN.md):
  // the browser calls NEXT_PUBLIC_API_BASE_URL directly — no Next API routes.
  reactStrictMode: true,
  // Keep Vercel's build trace scoped to the frontend rather than the monorepo
  // root (which also contains large training data and Python artifacts).
  outputFileTracingRoot: process.cwd(),
};

export default nextConfig;
