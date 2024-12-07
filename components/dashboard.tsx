"use client"

import { useState, useEffect } from 'react';
import { useGamification } from '@/contexts/gamification-context';
import { Trophy, Star, Target, Brain, TrendingUp, Award, Zap, BookOpen, Users, BarChart } from 'lucide-react';
import { Progress } from "@/components/ui/progress"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"
import { Flashcards } from './flashcards'
import { Analytics } from './analytics'
import { PracticeTest } from './practice-test';
import CARSPractice from './cars-practice'; // Import the CARSPractice component
import { XP_RULES, DAILY_GOALS, LEVEL_THRESHOLDS } from '@/utils/gamification';

export function Dashboard() {
  const { state, addXP, reviewFlashcard, completePracticeTest, updateStudyTime } = useGamification();
  const [selectedSection, setSelectedSection] = useState('dashboard');

  useEffect(() => {
    console.log('Gamification state updated:', state);
  }, [state]);

  // Calculate daily progress percentage
  const dailyProgressPercentage = Math.max(0, Math.min(100, 
    ((state.dailyStats?.dailyXp || 0) / DAILY_GOALS.MIN_XP) * 100
  ));

  const handleStatsUpdate = ({ xp, correct, total }) => {
    const accuracy = (correct / total) * 100;
    const xpEarned = Math.floor(XP_RULES.PRACTICE_TEST_COMPLETION * (accuracy / 100));
    addXP(xpEarned);
  };

  const renderSection = () => {
    switch (selectedSection) {
      case 'flashcards':
        return <Flashcards />;
      case 'analytics':
        return <Analytics />;
      case 'dashboard':
      default:
        return renderDashboard();
      case 'practice-test':
        return <PracticeTest 
          onBackToDashboard={() => setSelectedSection('dashboard')} 
          onUpdateStats={handleStatsUpdate}
          addXP={addXP}
        />;
    }
  };

  const renderDashboard = () => (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Study Progress Card */}
      <Card className="bg-gray-800 border-0">
        <CardHeader>
          <CardTitle className="text-xl font-bold text-white">Study Progress</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {[
            { subject: 'Biology', progress: Math.round((state.dailyStats.flashcardsReviewed / 100) * 100), color: 'bg-green-500' },
            { subject: 'Chemistry', progress: Math.round((state.dailyStats.practiceQuestions / 50) * 100), color: 'bg-blue-500' },
            { subject: 'Physics', progress: Math.round((state.dailyStats.studyMinutes / 120) * 100), color: 'bg-purple-500' },
            { subject: 'CARS', progress: Math.round((state.dailyStats.dailyXp / 500) * 100), color: 'bg-yellow-500' }
          ].map(subject => (
            <div key={subject.subject}>
              <div className="flex justify-between items-center mb-1">
                <span className="text-white">{subject.subject}</span>
                <span className="text-sm text-gray-400">{Math.min(subject.progress, 100)}%</span>
              </div>
              <Progress value={Math.min(subject.progress, 100)} className="h-2 bg-gray-700" indicatorClassName={subject.color} />
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Achievements Card */}
      <Card className="bg-gray-800 border-0">
        <CardHeader>
          <CardTitle className="text-xl font-bold text-white">Recent Achievements</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {[
            { name: 'Biology Master', desc: 'Complete 100 Biology cards', icon: Brain, color: 'text-green-500' },
            { name: 'Streak Master', desc: '10 day study streak', icon: Target, color: 'text-red-500' },
            { name: 'Quiz Champion', desc: 'Score 90%+ on 5 quizzes', icon: Trophy, color: 'text-yellow-500' }
          ].map(achievement => (
            <motion.div 
              key={achievement.name}
              whileHover={{ scale: 1.02 }}
              className="flex items-center space-x-3 p-3 bg-gray-700 rounded-lg"
            >
              <achievement.icon size={24} className={achievement.color} />
              <div>
                <div className="font-medium text-white">{achievement.name}</div>
                <div className="text-sm text-gray-400">{achievement.desc}</div>
              </div>
            </motion.div>
          ))}
        </CardContent>
      </Card>

      {/* Study Path Card */}
      <Card className="bg-gray-800 border-0">
        <CardHeader>
          <CardTitle className="text-xl font-bold text-white">Optimized Study Path</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {[
            { topic: 'Cellular Respiration', time: '45min', priority: 'High', score: 85 },
            { topic: 'Thermodynamics', time: '30min', priority: 'Medium', score: 72 },
            { topic: 'CARS Passage 3', time: '60min', priority: 'High', score: 68 }
          ].map(path => (
            <motion.div 
              key={path.topic}
              whileHover={{ scale: 1.02 }}
              className="p-3 bg-gray-700 rounded-lg"
            >
              <div className="flex justify-between items-center">
                <span className="font-medium text-white">{path.topic}</span>
                <span className={`text-sm px-2 py-1 rounded ${
                  path.priority === 'High' ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'
                }`}>
                  {path.priority}
                </span>
              </div>
              <div className="flex justify-between items-center mt-2 text-sm text-gray-400">
                <span>{path.time}</span>
                <span>Previous: {path.score}%</span>
              </div>
            </motion.div>
          ))}
        </CardContent>
      </Card>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Top Stats Bar */}
      <div className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-8">
            {/* Level Progress */}
            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="w-12 h-12 rounded-full bg-blue-500 flex items-center justify-center text-lg font-bold">
                {state.level}
                </div>
                <div className="absolute -top-1 -right-1 bg-yellow-500 text-black text-xs font-bold rounded-full w-6 h-6 flex items-center justify-center">
                  <Star size={12} />
                </div>
              </div>
              <div>
                <div className="text-sm text-gray-400">Level {state.level}</div>
                <Progress 
                  value={(state.xp % 1000) / 10} 
                  className="w-32 h-2 bg-gray-700" 
                  indicatorClassName="bg-blue-500" 
                />
              </div>
            </div>

            {/* XP Counter */}
            <div className="flex items-center space-x-2">
              <Zap size={20} className="text-yellow-500" />
              <span className="font-bold">
                {state.xp.toLocaleString()} / {LEVEL_THRESHOLDS[state.level - 1]} XP
              </span>
            </div>

            {/* Streak */}
            <div className="flex items-center space-x-2">
              <Target size={20} className="text-red-500" />
              <span className="font-bold">{state.streakDays} Day Streak</span>
            </div>
          </div>

          {/* Today's Goal */}
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-400">Today's Goal</div>
            <Progress 
              value={dailyProgressPercentage}
              className="w-40 h-3 bg-gray-700" 
              indicatorClassName="bg-green-500" 
            />
            <div className="text-sm">
              {Math.floor(dailyProgressPercentage)}%
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="max-w-7xl mx-auto p-6">
        {renderSection()}

        {/* Interactive Modes Bar */}
        <div className="mt-6 grid grid-cols-4 gap-4">
          {[
            { name: 'Flash Cards', icon: BookOpen, color: 'bg-blue-500', section: 'flashcards' },
            { name: 'Group Study', icon: Users, color: 'bg-green-500', section: 'groupstudy' },
            { name: 'Practice Test', icon: Brain, color: 'bg-purple-500', section: 'practice-test' },
            { name: 'Analytics', icon: BarChart, color: 'bg-yellow-500', section: 'analytics' }
          ].map(mode => (
            <Button
              key={mode.name}
              className={`${mode.color} p-4 rounded-xl flex items-center justify-center space-x-2 hover:opacity-90 transition-opacity`}
              onClick={() => setSelectedSection(mode.section)}
            >
              <mode.icon size={20} />
              <span>{mode.name}</span>
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}

