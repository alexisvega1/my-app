"use client"

import { useState } from 'react';
import { useChat } from 'ai/react';
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Loader2, Send } from 'lucide-react';

interface MedTutorProps {
  addXP: (amount: number) => void;
}

export default function MedTutor({ addXP }: MedTutorProps) {
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: '/api/med-tutor',
  });

  const handleFormSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    handleSubmit(e);
    // Add XP for each question asked
    addXP(5);
  };

  return (
    <Card className="bg-gray-800 border-0">
      <CardHeader>
        <CardTitle className="text-2xl font-bold text-white">MedTutor</CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[400px] mb-4 p-4 rounded-md bg-gray-700">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`mb-4 ${
                message.role === 'user' ? 'text-blue-400' : 'text-green-400'
              }`}
            >
              <strong>{message.role === 'user' ? 'You: ' : 'MedTutor: '}</strong>
              {message.content}
            </div>
          ))}
          {isLoading && (
            <div className="text-yellow-400">
              <Loader2 className="w-4 h-4 animate-spin inline mr-2" />
              MedTutor is thinking...
            </div>
          )}
        </ScrollArea>
        <form onSubmit={handleFormSubmit} className="flex space-x-2">
          <Input
            value={input}
            onChange={handleInputChange}
            placeholder="Ask a question about MCAT prep..."
            className="flex-grow bg-gray-700 text-white border-gray-600"
          />
          <Button type="submit" disabled={isLoading}>
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}

