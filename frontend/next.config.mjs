/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static-friendly one-pager. Uploads go straight to FastAPI (PLAN.md):
  // the browser calls NEXT_PUBLIC_API_BASE_URL directly — no Next API routes.
  reactStrictMode: true,
};

export default nextConfig;
