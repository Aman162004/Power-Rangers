/**
 * User menu dropdown component
 */

import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { LogOut, User, Settings } from 'lucide-react';

export function UserMenu() {
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleLogout = async () => {
    await logout();
    navigate('/login', { replace: true });
  };

  if (!user) return null;

  const getRoleBadgeColor = (role: string) => {
    switch (role) {
      case 'ADMIN':
        return 'bg-red-500/20 text-red-200';
      case 'ANALYST':
        return 'bg-blue-500/20 text-blue-200';
      case 'OPERATOR':
        return 'bg-green-500/20 text-green-200';
      default:
        return 'bg-slate-500/20 text-slate-200';
    }
  };

  return (
    <div className="relative" ref={menuRef}>
      {/* Menu Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-3 px-4 py-2 bg-slate-700/40 hover:bg-slate-700/60 border border-slate-600 rounded-lg transition"
      >
        <div className="flex items-center gap-2">
          <User className="w-4 h-4 text-slate-400" />
          <div className="text-left">
            <div className="text-sm font-medium text-white">{user.username}</div>
            <div className={`text-xs px-2 py-0.5 rounded ${getRoleBadgeColor(user.role)}`}>
              {user.role}
            </div>
          </div>
        </div>
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div className="absolute right-0 mt-2 w-48 bg-slate-800 border border-slate-700 rounded-lg shadow-xl z-50 overflow-hidden">
          {/* User Info */}
          <div className="px-4 py-3 border-b border-slate-700 bg-slate-900/50">
            <p className="text-sm font-medium text-white">{user.full_name || user.username}</p>
            <p className="text-xs text-slate-400">{user.email}</p>
          </div>

          {/* Menu Items */}
          <div className="py-2">
            {user.role === 'ADMIN' && (
              <button
                onClick={() => {
                  navigate('/admin');
                  setIsOpen(false);
                }}
                className="w-full text-left px-4 py-2 text-sm text-slate-300 hover:bg-slate-700 transition flex items-center gap-2"
              >
                <Settings className="w-4 h-4" />
                Admin Panel
              </button>
            )}

            <button
              onClick={handleLogout}
              className="w-full text-left px-4 py-2 text-sm text-red-300 hover:bg-red-500/10 transition flex items-center gap-2"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default UserMenu;
