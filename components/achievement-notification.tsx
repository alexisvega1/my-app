"use client"

import { motion, AnimatePresence } from 'framer-motion';
import { Award } from 'lucide-react';

interface AchievementNotificationProps {
  title: string;
  description: string;
  xpReward: number;
  onClose: () => void;
}

export function AchievementNotification({
  title,
  description,
  xpReward,
  onClose,
}: AchievementNotificationProps) {
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -50 }}
        className="fixed bottom-4 right-4 z-50"
      >
        <div className="flex items-center space-x-4 rounded-lg bg-gray-800 p-4 shadow-lg">
          <div className="rounded-full bg-yellow-400 p-2">
            <Award className="h-6 w-6 text-gray-900" />
          </div>
          <div>
            <h3 className="font-bold text-white">{title}</h3>
            <p className="text-sm text-gray-300">{description}</p>
            <p className="mt-1 text-sm font-semibold text-yellow-400">
              +{xpReward} XP
            </p>
          </div>
          <button
            onClick={onClose}
            className="ml-4 rounded-full p-1 text-gray-400 hover:bg-gray-700 hover:text-white"
          >
            ×
          </button>
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

