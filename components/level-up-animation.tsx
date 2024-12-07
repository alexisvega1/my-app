"use client"

import { motion, AnimatePresence } from 'framer-motion';
import { Star, Trophy } from 'lucide-react';
import { useEffect, useState } from 'react';

interface LevelUpAnimationProps {
  level: number;
  onComplete?: () => void;
}

export function LevelUpAnimation({ level, onComplete }: LevelUpAnimationProps) {
  const [show, setShow] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShow(false);
      onComplete?.();
    }, 3000);

    return () => clearTimeout(timer);
  }, [onComplete]);

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 1.5 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
        >
          <motion.div
            initial={{ y: 50 }}
            animate={{ y: 0 }}
            className="relative flex flex-col items-center"
          >
            <motion.div
              animate={{
                rotate: [0, 360],
                scale: [1, 1.2, 1],
              }}
              transition={{
                duration: 1.5,
                ease: "easeInOut",
                times: [0, 0.5, 1],
                repeat: Infinity,
              }}
              className="absolute -top-12 text-yellow-400"
            >
              <Star className="h-12 w-12" />
            </motion.div>
            
            <motion.div
              animate={{
                y: [0, -20, 0],
              }}
              transition={{
                duration: 2,
                ease: "easeInOut",
                times: [0, 0.5, 1],
                repeat: Infinity,
              }}
              className="flex flex-col items-center space-y-4 rounded-lg bg-gray-800 p-8 shadow-xl"
            >
              <Trophy className="h-16 w-16 text-yellow-400" />
              <h2 className="text-3xl font-bold text-white">Level Up!</h2>
              <p className="text-xl text-gray-300">
                You've reached level {level}
              </p>
              <div className="mt-4 flex space-x-2">
                {Array.from({ length: 3 }).map((_, i) => (
                  <motion.span
                    key={i}
                    animate={{
                      scale: [1, 1.5, 1],
                      opacity: [0.5, 1, 0.5],
                    }}
                    transition={{
                      duration: 1,
                      ease: "easeInOut",
                      delay: i * 0.2,
                      repeat: Infinity,
                    }}
                    className="inline-block h-3 w-3 rounded-full bg-yellow-400"
                  />
                ))}
              </div>
            </motion.div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

