/**
 * Login page component
 */

import { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { AlertCircle } from 'lucide-react';

export function LoginPage() {
  const navigate = useNavigate();
  const { login, isLoading, isAuthenticated } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (isAuthenticated) {
      navigate('/', { replace: true });
    }
  }, [isAuthenticated, navigate]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!username || !password) {
      setError('Username and password are required');
      return;
    }

    try {
      await login(username, password);
    } catch (err: any) {
      setError(err || 'Login failed. Please try again.');
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden bg-gradient-to-br from-[#020617] via-[#0a1f3d] to-[#111827] flex items-center justify-center px-4">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(34,211,238,0.14),transparent_36%),radial-gradient(circle_at_80%_30%,rgba(16,185,129,0.12),transparent_35%)]" />
      <div className="w-full max-w-md">
        {/* Header */}
        <div className="text-center mb-8">
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-300 mb-3">Secure Access</p>
          <h1 className="text-3xl font-bold text-white mb-2">Power Rangers</h1>
          <p className="text-slate-300">Intelligence Grid Load Forecasting</p>
        </div>

        {/* Card */}
        <div className="relative z-10 rounded-2xl border border-cyan-300/15 bg-slate-900/55 p-8 shadow-[0_0_60px_rgba(8,145,178,0.14)] backdrop-blur">
          <h2 className="text-2xl font-semibold text-white mb-6">Sign In</h2>

          {/* Error Alert */}
          {error && (
            <div className="mb-6 p-3 bg-red-500/10 border border-red-500/50 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-500 flex-shrink-0 mt-0.5" />
              <p className="text-red-200 text-sm">{error}</p>
            </div>
          )}

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Username */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Username
              </label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter your username"
                className="w-full px-4 py-2.5 bg-slate-900/70 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-400/60 focus:border-cyan-300/60 transition"
                disabled={isLoading}
              />
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-2">
                Password
              </label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter your password"
                className="w-full px-4 py-2.5 bg-slate-900/70 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-400/60 focus:border-cyan-300/60 transition"
                disabled={isLoading}
              />
            </div>

            {/* Submit Button */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full mt-6 px-4 py-2.5 rounded-lg border border-cyan-300/25 bg-cyan-500/20 text-cyan-100 font-medium transition duration-200 hover:bg-cyan-500/30 disabled:bg-slate-700/60 disabled:text-slate-300 disabled:cursor-not-allowed"
            >
              {isLoading ? 'Signing in...' : 'Sign In'}
            </button>
          </form>

          {/* Demo Credentials Info */}
          <div className="mt-6 p-3 bg-slate-950/35 border border-cyan-300/20 rounded-lg">
            <p className="text-sm font-semibold text-cyan-200">Demo Credentials:</p>
            <div className="mt-2 space-y-2 text-sm text-slate-200">
              <p>
                <span className="font-medium">Admin:</span>{' '}
                <code className="bg-slate-700 px-2 py-1 rounded">admin</code> /{' '}
                <code className="bg-slate-700 px-2 py-1 rounded">changeme123!</code>
              </p>
              <p>
                <span className="font-medium">Energy Analyst:</span>{' '}
                <code className="bg-slate-700 px-2 py-1 rounded">analyst_demo</code> /{' '}
                <code className="bg-slate-700 px-2 py-1 rounded">analyst123!</code>
              </p>
              <p>
                <span className="font-medium">Power Grid Operator:</span>{' '}
                <code className="bg-slate-700 px-2 py-1 rounded">operator_demo</code> /{' '}
                <code className="bg-slate-700 px-2 py-1 rounded">operator123!</code>
              </p>
            </div>
          </div>

          {/* Register Link */}
          <div className="mt-6 text-center text-sm text-slate-400">
            Don't have an account?{' '}
            <Link to="/register" className="text-cyan-300 hover:text-cyan-200 font-medium">
              Register with invite token
            </Link>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-xs text-slate-400">
          Protected by JWT authentication
        </div>
      </div>
    </div>
  );
}

export default LoginPage;
