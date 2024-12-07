"use client"

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { motion } from "framer-motion";
import { ChevronRight, CheckCircle, AlertCircle } from 'lucide-react';
import { Progress } from "@/components/ui/progress";

const mockScoreData = [
  { subject: 'Biology', score: 85, fullMark: 100 },
  { subject: 'Chemistry', score: 78, fullMark: 100 },
  { subject: 'Physics', score: 72, fullMark: 100 },
  { subject: 'Psychology', score: 90, fullMark: 100 },
  { subject: 'CARS', score: 82, fullMark: 100 },
]

const mockTimeData = [
  { name: 'Week 1', Biology: 120, Chemistry: 90, Physics: 60, Psychology: 100, CARS: 80 },
  { name: 'Week 2', Biology: 150, Chemistry: 110, Physics: 80, Psychology: 120, CARS: 100 },
  { name: 'Week 3', Biology: 180, Chemistry: 130, Physics: 100, Psychology: 140, CARS: 120 },
  { name: 'Week 4', Biology: 200, Chemistry: 160, Physics: 120, Psychology: 160, CARS: 140 },
]

const mockProgressData = [
  { name: 'Jan', score: 450 },
  { name: 'Feb', score: 480 },
  { name: 'Mar', score: 510 },
  { name: 'Apr', score: 535 },
  { name: 'May', score: 560 },
  { name: 'Jun', score: 590 },
]

const topMedSchools = [
  { name: "Harvard", mcat: 520, gpa: 3.9, rank: 1, researchEmphasis: 0.9, clinicalEmphasis: 0.8, leadershipEmphasis: 0.7 },
  { name: "Johns Hopkins", mcat: 521, gpa: 3.91, rank: 2, researchEmphasis: 0.95, clinicalEmphasis: 0.75, leadershipEmphasis: 0.8 },
  { name: "Stanford", mcat: 519, gpa: 3.89, rank: 3, researchEmphasis: 0.85, clinicalEmphasis: 0.85, leadershipEmphasis: 0.75 },
  { name: "UPenn", mcat: 521, gpa: 3.89, rank: 4, researchEmphasis: 0.8, clinicalEmphasis: 0.9, leadershipEmphasis: 0.7 },
  { name: "NYU", mcat: 522, gpa: 3.96, rank: 5, researchEmphasis: 0.75, clinicalEmphasis: 0.95, leadershipEmphasis: 0.8 },
  { name: "Columbia", mcat: 521, gpa: 3.91, rank: 6, researchEmphasis: 0.85, clinicalEmphasis: 0.8, leadershipEmphasis: 0.75 },
  { name: "Mayo Clinic", mcat: 520, gpa: 3.92, rank: 7, researchEmphasis: 0.8, clinicalEmphasis: 0.9, leadershipEmphasis: 0.7 },
  { name: "UCLA", mcat: 517, gpa: 3.85, rank: 8, researchEmphasis: 0.75, clinicalEmphasis: 0.85, leadershipEmphasis: 0.8 },
  { name: "Wash U", mcat: 521, gpa: 3.91, rank: 9, researchEmphasis: 0.9, clinicalEmphasis: 0.75, leadershipEmphasis: 0.75 },
  { name: "Yale", mcat: 519, gpa: 3.89, rank: 10, researchEmphasis: 0.85, clinicalEmphasis: 0.8, leadershipEmphasis: 0.85 }
];

const medSchoolGPA = [
  { name: "Harvard", gpa: 3.9, rank: 1 },
  { name: "Johns Hopkins", gpa: 3.91, rank: 2 },
  { name: "Stanford", gpa: 3.89, rank: 3 },
  { name: "UPenn", gpa: 3.89, rank: 4 },
  { name: "NYU", gpa: 3.96, rank: 5 },
]

const mockInterviewData = {
  readyScore: 78,
  experienceStories: [
    { name: 'Clinical Experience', completed: true },
    { name: 'Research Impact', completed: true },
    { name: 'Leadership Challenge', completed: true },
    { name: 'Ethical Dilemma', completed: true }
  ],
  knowledgeAreas: [
    { name: 'Medical Ethics', completed: true },
    { name: 'Healthcare System', completed: true },
    { name: 'Current Events', completed: false },
    { name: 'Research Methods', completed: true }
  ],
  communicationSkills: [
    { name: 'Articulation', score: 85 },
    { name: 'Active Listening', score: 90 },
    { name: 'Non-verbal', score: 75 },
    { name: 'Empathy', score: 88 }
  ]
}

const mockActionPlan = {
  next30Days: [
    { task: 'Increase clinical volunteering hours', deadline: '2 weeks', impact: 'High Impact' },
    { task: 'Complete research manuscript', deadline: '4 weeks', impact: 'High Impact' }
  ],
  next2To3Months: [
    { task: 'Obtain additional shadowing hours', deadline: '8 weeks', impact: 'Medium Impact' },
    { task: 'Leadership role in service organization', deadline: '12 weeks', impact: 'Medium Impact' }
  ]
}

const mockCompetitiveness = {
  overallScore: 82,
  categories: [
    { name: 'Academic', score: 85, color: 'bg-blue-500' },
    { name: 'Clinical', score: 78, color: 'bg-blue-500' },
    { name: 'Research', score: 92, color: 'bg-green-500' },
    { name: 'Leadership', score: 70, color: 'bg-yellow-500' },
    { name: 'Service', score: 88, color: 'bg-blue-500' }
  ],
  strengths: [
    { 
      title: 'Research Experience',
      description: '2 publications, 3 years lab work',
      percentile: '92th percentile',
      color: 'bg-green-500'
    },
    {
      title: 'Clinical Experience',
      description: '1000+ hours, direct patient care',
      percentile: '85th percentile',
      color: 'bg-blue-500'
    },
    {
      title: 'Leadership Roles',
      description: '3 positions, project management',
      percentile: '78th percentile',
      color: 'bg-yellow-500'
    }
  ],
  improvements: [
    {
      area: 'Volunteer Service',
      current: '200 hours',
      target: '400+ hours',
      priority: 'High Priority'
    },
    {
      area: 'Shadowing Experience',
      current: '40 hours',
      target: '100+ hours',
      priority: 'High Priority'
    },
    {
      area: 'Leadership Activities',
      current: '2 positions',
      target: '3-4 positions',
      priority: 'Medium Priority'
    }
  ]
}

const mockPracticeQuestions = [
  'Why medicine?',
  'Challenging experience',
  'Research impact',
  'Healthcare changes'
]

const mockComparisonData = [
  {
    name: 'Jan',
    score: 450,
    topPerformers: 490,
    averageUsers: 470,
  },
  {
    name: 'Feb',
    score: 480,
    topPerformers: 505,
    averageUsers: 485,
  },
  {
    name: 'Mar',
    score: 510,
    topPerformers: 520,
    averageUsers: 500,
  },
  {
    name: 'Apr',
    score: 535,
    topPerformers: 530,
    averageUsers: 515,
  },
  {
    name: 'May',
    score: 560,
    topPerformers: 545,
    averageUsers: 525,
  },
  {
    name: 'Jun',
    score: 590,
    topPerformers: 560,
    averageUsers: 535,
  },
]

export function Analytics({ onBackToDashboard }) {
  const [activeTab, setActiveTab] = useState('overview');
  const [admissionPredictions, setAdmissionPredictions] = useState([]);
  const [selectedSchool, setSelectedSchool] = useState("all");
  const [showTopPerformers, setShowTopPerformers] = useState(false);
  const [showAverageUsers, setShowAverageUsers] = useState(false);

  const predictAdmission = (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);
    const mcatScore = parseInt(formData.get('mcatScore'));
    const gpa = parseFloat(formData.get('gpa'));
    const researchYears = parseFloat(formData.get('researchYears'));
    const publications = parseInt(formData.get('publications'));
    const volunteerHours = parseInt(formData.get('volunteerHours'));
    const clinicalHours = parseInt(formData.get('clinicalHours'));
    const shadowingHours = parseInt(formData.get('shadowingHours'));
    const leadershipYears = parseFloat(formData.get('leadershipYears'));
    
    const predictions = topMedSchools.map(school => {
      const mcatDiff = (mcatScore - school.mcat) / 2;
      const gpaDiff = (gpa - school.gpa) * 10;
      
      const researchScore = (researchYears * 5 + publications * 3) * school.researchEmphasis;
      
      const clinicalScore = ((clinicalHours / 100) + (shadowingHours / 50)) * school.clinicalEmphasis;
      
      const leadershipScore = (leadershipYears * 5 + (volunteerHours / 100)) * school.leadershipEmphasis;
      
      let chance = 50 + mcatDiff + gpaDiff + (researchScore + clinicalScore + leadershipScore) / 3;
      chance = Math.max(0, Math.min(100, chance));
      
      return {
        school: school.name,
        chance: chance.toFixed(1)
      };
    });

    setAdmissionPredictions(predictions);
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return (
          <div className="grid gap-6 md:grid-cols-2">
            <Card className="bg-gray-800 border-0">
              <CardHeader>
                <CardTitle className="text-white">Subject Performance</CardTitle>
                <CardDescription>Your scores across different MCAT subjects</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={mockScoreData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="subject" />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} />
                    <Radar name="Score" dataKey="score" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
                  </RadarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
            <Card className="bg-gray-800 border-0">
              <CardHeader>
                <CardTitle className="text-white">Weekly Study Time</CardTitle>
                <CardDescription>Hours spent studying each subject per week</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={mockTimeData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area type="monotone" dataKey="Biology" stackId="1" stroke="#8884d8" fill="#8884d8" />
                    <Area type="monotone" dataKey="Chemistry" stackId="1" stroke="#82ca9d" fill="#82ca9d" />
                    <Area type="monotone" dataKey="Physics" stackId="1" stroke="#ffc658" fill="#ffc658" />
                    <Area type="monotone" dataKey="Psychology" stackId="1" stroke="#ff8042" fill="#ff8042" />
                    <Area type="monotone" dataKey="CARS" stackId="1" stroke="#a4de6c" fill="#a4de6c" />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        );
      case 'prediction':
        return (
          <Card className="bg-gray-800 border-0">
            <CardHeader>
              <CardTitle className="text-white">Medical School Admission Prediction</CardTitle>
              <CardDescription>Estimate your chances based on your complete profile</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={predictAdmission} className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="mcatScore" className="text-white">MCAT Score (472-528)</Label>
                    <Input type="number" id="mcatScore" name="mcatScore" min="472" max="528" required className="bg-gray-700 border-gray-600 text-white" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="gpa" className="text-white">GPA (0.0-4.0)</Label>
                    <Input type="number" id="gpa" name="gpa" min="0" max="4" step="0.01" required className="bg-gray-700 border-gray-600 text-white" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="researchYears" className="text-white">Years of Research Experience</Label>
                    <Input type="number" id="researchYears" name="researchYears" min="0" max="10" step="0.5" required className="bg-gray-700 border-gray-600 text-white" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="publications" className="text-white">Number of Publications</Label>
                    <Input type="number" id="publications" name="publications" min="0" required className="bg-gray-700 border-gray-600 text-white" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="volunteerHours" className="text-white">Volunteer Hours</Label>
                    <Input type="number" id="volunteerHours" name="volunteerHours" min="0" required className="bg-gray-700 border-gray-600 text-white" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="clinicalHours" className="text-white">Clinical Experience Hours</Label>
                    <Input type="number" id="clinicalHours" name="clinicalHours" min="0" required className="bg-gray-700 border-gray-600 text-white" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="shadowingHours" className="text-white">Shadowing Hours</Label>
                    <Input type="number" id="shadowingHours" name="shadowingHours" min="0" required className="bg-gray-700 border-gray-600 text-white" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="leadershipYears" className="text-white">Years of Leadership Experience</Label>
                    <Input type="number" id="leadershipYears" name="leadershipYears" min="0" max="10" step="0.5" required className="bg-gray-700 border-gray-600 text-white" />
                  </div>
                </div>
                <Button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 text-white">
                  Predict Admission Chances
                </Button>
              </form>
              {admissionPredictions.length > 0 && (
                <div className="mt-6">
                  <h3 className="text-lg font-semibold mb-4 text-white">Predicted Chances:</h3>
                  <div className="space-y-3">
                    {admissionPredictions.map((prediction, index) => (
                      <div key={index} className={`flex items-center justify-between p-3 rounded-lg ${
                        parseFloat(prediction.chance) >= 75 ? 'bg-green-500/20 text-green-100' :
                        parseFloat(prediction.chance) >= 70 ? 'bg-blue-500/20 text-blue-100' :
                        parseFloat(prediction.chance) >= 60 ? 'bg-yellow-500/20 text-yellow-100' :
                        'bg-red-500/20 text-red-100'
                      }`}>
                        <span className="font-medium">{prediction.school}</span>
                        <span className="font-bold">{prediction.chance}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        );
      case 'progress':
        return (
          <Card className="bg-gray-800 border-0">
            <CardHeader>
              <CardTitle className="text-xl text-white">MCAT Score Progress</CardTitle>
              <CardDescription className="text-gray-400">
                Your MCAT score improvement over time
              </CardDescription>
              <div className="flex flex-wrap gap-4 mt-4">
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="topPerformers"
                    checked={showTopPerformers}
                    onChange={(e) => setShowTopPerformers(e.target.checked)}
                    className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500"
                  />
                  <label htmlFor="topPerformers" className="text-sm text-gray-300">
                    Compare with Top Performers
                  </label>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="averageUsers"
                    checked={showAverageUsers}
                    onChange={(e) => setShowAverageUsers(e.target.checked)}
                    className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-blue-500"
                  />
                  <label htmlFor="averageUsers" className="text-sm text-gray-300">
                    Compare with Average Users
                  </label>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={mockComparisonData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid 
                      strokeDasharray="3 3" 
                      stroke="rgba(255,255,255,0.1)"
                    />
                    <XAxis 
                      dataKey="name" 
                      stroke="#94a3b8"
                      tick={{ fill: '#94a3b8' }}
                    />
                    <YAxis 
                      domain={[400, 600]} 
                      stroke="#94a3b8"
                      tick={{ fill: '#94a3b8' }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1f2937',
                        border: '1px solid #374151',
                        borderRadius: '0.5rem',
                        color: '#f3f4f6'
                      }}
                      itemStyle={{ color: '#f3f4f6' }}
                      labelStyle={{ color: '#94a3b8' }}
                    />
                    <Legend 
                      verticalAlign="top" 
                      height={36}
                      wrapperStyle={{ color: '#f3f4f6' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="score"
                      name="Your Score"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={{
                        r: 4,
                        fill: '#3b82f6',
                        strokeWidth: 2
                      }}
                      activeDot={{
                        r: 6,
                        fill: '#3b82f6',
                        stroke: '#fff',
                        strokeWidth: 2
                      }}
                    />
                    {showTopPerformers && (
                      <Line
                        type="monotone"
                        dataKey="topPerformers"
                        name="Top Performers"
                        stroke="#22c55e"
                        strokeWidth={2}
                        strokeDasharray="5 5"
                        dot={{
                          r: 4,
                          fill: '#22c55e',
                          strokeWidth: 2
                        }}
                      />
                    )}
                    {showAverageUsers && (
                      <Line
                        type="monotone"
                        dataKey="averageUsers"
                        name="Average Users"
                        stroke="#eab308"
                        strokeWidth={2}
                        strokeDasharray="3 3"
                        dot={{
                          r: 4,
                          fill: '#eab308',
                          strokeWidth: 2
                        }}
                      />
                    )}
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-6 p-4 bg-gray-700/50 rounded-lg">
                <h4 className="text-lg font-semibold text-white mb-2">Progress Insights</h4>
                <p className="text-gray-300">
                  Your MCAT score has improved by {mockComparisonData[mockComparisonData.length - 1].score - mockComparisonData[0].score} points since January. 
                  {showTopPerformers && " You're performing above the top performers average in recent months."}
                  {showAverageUsers && " You're consistently above the average user performance."}
                </p>
              </div>
            </CardContent>
          </Card>
        );
      case 'schools':
        return (
          <div className="grid gap-6 md:grid-cols-2">
            <Card className="bg-[#262b3d] border-0">
              <CardHeader>
                <CardTitle className="text-xl text-gray-100">Top 10 Schools - MCAT Scores</CardTitle>
                <CardDescription>Average MCAT scores for top medical schools</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart 
                    data={topMedSchools}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[510, 525]} />
                    <YAxis type="category" dataKey="name" width={150} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="mcat" fill="#8884d8" name="Avg MCAT" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="bg-[#262b3d] border-0">
              <CardHeader>
                <CardTitle className="text-xl text-gray-100">Top 10 Schools - GPA</CardTitle>
                <CardDescription>Average GPAs for top medical schools</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart 
                    data={topMedSchools}
                    layout="vertical"
                    margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" domain={[3.8, 4.0]} />
                    <YAxis type="category" dataKey="name" width={150} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="gpa" fill="#82ca9d" name="Avg GPA" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        );
      case 'profile':
        return (
          <>
            {renderProfileAnalysis()}
            {renderInterviewReadiness()}
            {renderActionPlan()}
            {renderPracticeQuestions()}
          </>
        );
      default:
        return null;
    }
  };

  const renderProfileAnalysis = () => (
    <Card className="bg-gray-800 border-0 mb-6">
      <CardHeader className="flex flex-row justify-between items-center">
        <div>
          <CardTitle className="text-xl text-white">Application Competitiveness</CardTitle>
          <CardDescription>Based on your complete profile analysis</CardDescription>
        </div>
        <div className="text-3xl font-bold text-blue-500">{mockCompetitiveness.overallScore}/100</div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-3">
          {mockCompetitiveness.categories.map((category, index) => (
            <div key={index}>
              <div className="flex justify-between text-sm text-white mb-1">
                <span>{category.name}</span>
                <span>{category.score}%</span>
              </div>
              <Progress value={category.score} className={`h-2 bg-gray-700`} indicatorClassName={category.color} />
            </div>
          ))}
        </div>

        <div>
          <h3 className="text-lg font-semibold text-white mb-3">Profile Strengths</h3>
          <div className="space-y-2">
            {mockCompetitiveness.strengths.map((strength, index) => (
              <div key={index} className="bg-gray-700 p-3 rounded-lg">
                <div className="flex justify-between items-center">
                  <span className="text-white font-medium">{strength.title}</span>
                  <span className={`px-2 py-1 ${strength.color}/20 text-sm rounded-full`}>
                    {strength.percentile}
                  </span>
                </div>
                <div className="text-sm text-gray-400 mt-1">{strength.description}</div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold text-white mb-3">Areas for Improvement</h3>
          <div className="space-y-2">
            {mockCompetitiveness.improvements.map((item, index) => (
              <div key={index} className="bg-gray-700 p-3 rounded-lg">
                <div className="flex justify-between items-center">
                  <span className="text-white">{item.area}</span>
                  <span className={`px-2 py-1 ${
                    item.priority === 'High Priority' ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'
                  } rounded-full text-sm`}>
                    {item.priority}
                  </span>
                </div>
                <div className="text-sm text-gray-400 mt-1">
                  Current: {item.current} → Target: {item.target}
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const renderInterviewReadiness = () => (
    <Card className="bg-gray-800 border-0 mb-6">
      <CardHeader className="flex flex-row justify-between items-center">
        <div>
          <CardTitle className="text-xl text-white">Interview Readiness</CardTitle>
          <CardDescription>Based on your profile and experiences</CardDescription>
        </div>
        <div className="text-5xl font-bold text-green-500">{mockInterviewData.readyScore}%</div>
      </CardHeader>
      <CardContent className="space-y-6">
        <div>
          <h3 className="text-lg font-semibold text-white mb-3">Experience Stories</h3>
          <div className="grid grid-cols-2 gap-2">
            {mockInterviewData.experienceStories.map((story, index) => (
              <div key={index} className="flex items-center space-x-2 text-white">
                <CheckCircle className={`w-4 h-4 ${story.completed ? 'text-green-500' : 'text-gray-600'}`} />
                <span>{story.name}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold text-white mb-3">Knowledge Areas</h3>
          <div className="grid grid-cols-2 gap-2">
            {mockInterviewData.knowledgeAreas.map((area, index) => (
              <div key={index} className="flex items-center space-x-2 text-white">
                {area.completed ? (
                  <CheckCircle className="w-4 h-4 text-green-500" />
                ) : (
                  <AlertCircle className="w-4 h-4 text-red-500" />
                )}
                <span>{area.name}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold text-white mb-3">Communication Skills</h3>
          <div className="space-y-3">
            {mockInterviewData.communicationSkills.map((skill, index) => (
              <div key={index}>
                <div className="flex justify-between text-sm text-white mb-1">
                  <span>{skill.name}</span>
                  <span>{skill.score}%</span>
                </div>
                <Progress value={skill.score} className="h-2 bg-gray-700" />
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const renderActionPlan = () => (
    <Card className="bg-gray-800 border-0 mb-6">
      <CardHeader>
        <CardTitle className="text-xl text-white">Action Plan</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div>
          <h3 className="text-lg font-semibold text-blue-400 mb-3">Next 30 Days</h3>
          <div className="space-y-2">
            {mockActionPlan.next30Days.map((item, index) => (
              <div key={index} className="bg-gray-700 p-3 rounded-lg">
                <div className="flex justify-between items-center">
                  <span className="text-white">{item.task}</span>
                  <span className="px-2 py-1 bg-red-500/20 text-red-400 rounded-full text-sm">
                    {item.impact}
                  </span>
                </div>
                <div className="text-sm text-gray-400 mt-1">Deadline: {item.deadline}</div>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h3 className="text-lg font-semibold text-blue-400 mb-3">2-3 Months</h3>
          <div className="space-y-2">
            {mockActionPlan.next2To3Months.map((item, index) => (
              <div key={index} className="bg-gray-700 p-3 rounded-lg">
                <div className="flex justify-between items-center">
                  <span className="text-white">{item.task}</span>
                  <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded-full text-sm">
                    {item.impact}
                  </span>
                </div>
                <div className="text-sm text-gray-400 mt-1">Deadline: {item.deadline}</div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const renderPracticeQuestions = () => (
    <Card className="bg-gray-800 border-0 mb-6">
      <CardHeader>
        <CardTitle className="text-xl text-white">Practice Questions</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {mockPracticeQuestions.map((question, index) => (
            <Button
              key={index}
              variant="ghost"
              className="w-full justify-between text-white hover:bg-gray-700"
            >
              {question}
              <ChevronRight className="h-4 w-4" />
            </Button>
          ))}
        </div>
      </CardContent>
    </Card>
  );

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <h1 className="text-3xl font-bold mb-6 text-white">Analytics Dashboard</h1>
      
      <div className="flex space-x-1 mb-6 bg-gray-800 p-1.5 rounded-lg overflow-x-auto">
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'progress', label: 'Progress' },
          { id: 'prediction', label: 'Admission Prediction' },
          { id: 'schools', label: 'Top Schools' },
          { id: 'profile', label: 'Profile Analysis' }
        ].map(tab => (
          <Button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            variant={activeTab === tab.id ? "default" : "ghost"}
            className={`flex-shrink-0 text-sm px-3 ${
              activeTab === tab.id 
                ? 'bg-blue-600 text-white' 
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            {tab.label}
          </Button>
        ))}
      </div>

      <div className="mt-6">
        {renderContent()}
      </div>
    </div>
  );
}

