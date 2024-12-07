"use client"

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts'
import { motion } from "framer-motion"

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
  { name: "Harvard Medical School", mcat: 520, gpa: 3.9, rank: 1, researchEmphasis: 0.9, clinicalEmphasis: 0.8, leadershipEmphasis: 0.7 },
  { name: "Johns Hopkins", mcat: 521, gpa: 3.91, rank: 2, researchEmphasis: 0.95, clinicalEmphasis: 0.75, leadershipEmphasis: 0.8 },
  { name: "Stanford", mcat: 519, gpa: 3.89, rank: 3, researchEmphasis: 0.85, clinicalEmphasis: 0.85, leadershipEmphasis: 0.75 },
  { name: "UPenn", mcat: 521, gpa: 3.89, rank: 4, researchEmphasis: 0.8, clinicalEmphasis: 0.9, leadershipEmphasis: 0.7 },
  { name: "NYU", mcat: 522, gpa: 3.96, rank: 5, researchEmphasis: 0.75, clinicalEmphasis: 0.95, leadershipEmphasis: 0.8 },
  { name: "Columbia", mcat: 521, gpa: 3.91, rank: 6, researchEmphasis: 0.85, clinicalEmphasis: 0.8, leadershipEmphasis: 0.75 },
  { name: "Mayo Clinic", mcat: 520, gpa: 3.92, rank: 7, researchEmphasis: 0.8, clinicalEmphasis: 0.9, leadershipEmphasis: 0.7 },
  { name: "UCLA", mcat: 517, gpa: 3.85, rank: 8, researchEmphasis: 0.75, clinicalEmphasis: 0.85, leadershipEmphasis: 0.8 },
  { name: "Washington University", mcat: 521, gpa: 3.91, rank: 9, researchEmphasis: 0.9, clinicalEmphasis: 0.75, leadershipEmphasis: 0.75 },
  { name: "Yale", mcat: 519, gpa: 3.89, rank: 10, researchEmphasis: 0.85, clinicalEmphasis: 0.8, leadershipEmphasis: 0.85 }
];

const medSchoolGPA = [
  { name: "Harvard", gpa: 3.9, rank: 1 },
  { name: "Johns Hopkins", gpa: 3.91, rank: 2 },
  { name: "Stanford", gpa: 3.89, rank: 3 },
  { name: "UPenn", gpa: 3.89, rank: 4 },
  { name: "NYU", gpa: 3.96, rank: 5 },
]

export function Analytics() {
  const [activeTab, setActiveTab] = useState('overview')
  const [admissionPredictions, setAdmissionPredictions] = useState([])
  const [selectedSchool, setSelectedSchool] = useState("all")

  const predictAdmission = (event) => {
    event.preventDefault()
    const formData = new FormData(event.target)
    const mcatScore = parseInt(formData.get('mcatScore'))
    const gpa = parseFloat(formData.get('gpa'))
    const researchYears = parseFloat(formData.get('researchYears'))
    const publications = parseInt(formData.get('publications'))
    const volunteerHours = parseInt(formData.get('volunteerHours'))
    const clinicalHours = parseInt(formData.get('clinicalHours'))
    const shadowingHours = parseInt(formData.get('shadowingHours'))
    const leadershipYears = parseFloat(formData.get('leadershipYears'))
    
    const predictions = topMedSchools.map(school => {
      // Base score from MCAT and GPA
      const mcatDiff = (mcatScore - school.mcat) / 2
      const gpaDiff = (gpa - school.gpa) * 10
      
      // Research score (weighted by school's emphasis)
      const researchScore = (researchYears * 5 + publications * 3) * school.researchEmphasis
      
      // Clinical experience score
      const clinicalScore = ((clinicalHours / 100) + (shadowingHours / 50)) * school.clinicalEmphasis
      
      // Leadership and service score
      const leadershipScore = (leadershipYears * 5 + (volunteerHours / 100)) * school.leadershipEmphasis
      
      // Calculate final chance
      let chance = 50 + mcatDiff + gpaDiff + (researchScore + clinicalScore + leadershipScore) / 3
      chance = Math.max(0, Math.min(100, chance))
      
      return {
        school: school.name,
        chance: chance.toFixed(1)
      }
    })

    setAdmissionPredictions(predictions)
  }

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
                      <div key={index} className="flex items-center justify-between bg-gray-700 p-3 rounded-lg">
                        <span className="text-white">{prediction.school}</span>
                        <span className="text-white font-bold">{prediction.chance}%</span>
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
          <Card>
            <CardHeader>
              <CardTitle>MCAT Score Progress</CardTitle>
              <CardDescription>Your MCAT score improvement over time</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={mockProgressData}
                           margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis domain={[400, 600]} />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="score" stroke="#8884d8" activeDot={{ r: 8 }} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        )
      case 'schools':
        return (
          <div className="grid gap-6 md:grid-cols-2">
            <Card className="bg-[#262b3d] border-0">
              <CardHeader>
                <CardTitle className="text-xl text-gray-100">Top 5 Schools - MCAT Scores</CardTitle>
                <CardDescription>Average MCAT scores for top medical schools</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={topMedSchools}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[500, 525]} />
                    <Tooltip />
                    <Bar dataKey="mcat" fill="#8884d8" name="Avg MCAT" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="bg-[#262b3d] border-0">
              <CardHeader>
                <CardTitle className="text-xl text-gray-100">Top 5 Schools - GPA</CardTitle>
                <CardDescription>Average GPAs for top medical schools</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={medSchoolGPA}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[3.5, 4.0]} />
                    <Tooltip />
                    <Bar dataKey="gpa" fill="#82ca9d" name="Avg GPA" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        )
      default:
        return null;
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <h1 className="text-3xl font-bold mb-6 text-white">Analytics Dashboard</h1>
      
      {/* Navigation */}
      <div className="flex space-x-2 mb-6 bg-gray-800 p-2 rounded-lg overflow-x-auto">
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'progress', label: 'Progress' },
          { id: 'prediction', label: 'Admission Prediction' },
          { id: 'schools', label: 'Top Schools' }
        ].map(tab => (
          <Button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            variant={activeTab === tab.id ? "default" : "ghost"}
            className={`flex-shrink-0 ${
              activeTab === tab.id 
                ? 'bg-blue-600 text-white' 
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            {tab.label}
          </Button>
        ))}
      </div>

      {/* Content */}
      <div className="mt-6">
        {renderContent()}
      </div>
    </div>
  )
}

