"use client"

import { useState, useEffect } from 'react';
import { Clock, CheckCircle, AlertCircle, ChevronLeft, ChevronRight, ArrowLeft } from 'lucide-react';
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import CARSPractice from './cars-practice';
import TimeAttackMode from './time-attack-mode';
import MedTutor from './med-tutor';

const testData = {
  totalQuestions: 30,
  section: 'Biological and Biochemical Foundations of Living Systems',
  timeLimit: 20, // minutes
  questions: [
    {
      id: 1,
      passage: "Enzymes are crucial biological catalysts that facilitate various chemical reactions in living organisms.",
      question: "What are the 6 classes of Enzymes?",
      options: [
        "1. Ligase 2. Isomerase 3. Lyase 4. Hydrolyase 5. Oxioreductase 6. Transferase",
        "1. Protease 2. Lipase 3. Amylase 4. Cellulase 5. Nuclease 6. Phosphatase",
        "1. Oxidase 2. Reductase 3. Hydrolase 4. Synthetase 5. Kinase 6. Phosphorylase",
        "1. Dehydrogenase 2. Carboxylase 3. Decarboxylase 4. Transaminase 5. Peptidase 6. Esterase"
      ],
      correct: 0,
      explanation: "The 6 classes of Enzymes are: 1. Ligase 2. Isomerase 3. Lyase 4. Hydrolyase 5. Oxioreductase 6. Transferase. A helpful mnemonic to remember this is LIL HOT.",
      difficulty: 'medium',
      topics: ['Enzymes', 'Biochemistry']
    },
    {
      id: 2,
      passage: "Carbohydrates are a class of biological molecules with various functions in living organisms.",
      question: "What are the types of monosaccharides based on their carbon count?",
      options: [
        "Monoses, dioses, trioses, and tetroses",
        "Trioses, tetroses, pentoses, and hexoses",
        "Dioses, trioses, tetroses, and pentoses",
        "Monoses, dioses, trioses, and pentoses"
      ],
      correct: 1,
      explanation: "Monosaccharides can have 3 carbons (trioses), 4 carbons (tetroses), 5 carbons (pentoses), or 6 carbons (hexoses).",
      difficulty: 'medium',
      topics: ['Carbohydrates', 'Biochemistry']
    },
    {
      id: 3,
      passage: "Lipids are a diverse group of biological molecules with various functions in living organisms.",
      question: "What are the three main types of lipids?",
      options: [
        "Structural Lipids, Signalling Lipids, and Energy Storage Lipids",
        "Fatty Acids, Glycerol, and Cholesterol",
        "Phospholipids, Triglycerides, and Steroids",
        "Saturated Fats, Unsaturated Fats, and Trans Fats"
      ],
      correct: 0,
      explanation: "The three main types of lipids are: 1. Structural Lipids 2. Signalling Lipids 3. Energy Storage Lipids",
      difficulty: 'medium',
      topics: ['Lipids', 'Biochemistry']
    },
    {
      id: 4,
      passage: "Proteins play crucial roles in the structure and function of living organisms.",
      question: "What are the primary structural proteins in the body?",
      options: [
        "Collagen, Elastin, Keratin, Actin, Tubulin",
        "Hemoglobin, Myoglobin, Albumin, Fibrinogen, Globulin",
        "Insulin, Glucagon, Growth Hormone, Thyroxine, Cortisol",
        "Amylase, Lipase, Pepsin, Trypsin, Chymotrypsin"
      ],
      correct: 0,
      explanation: "The primary structural proteins in the body are: 1. Collagen 2. Elastin 3. Keratin 4. Actin 5. Tubulin. A helpful mnemonic to remember this is: Cold Elephants Kill Angry Tigers.",
      difficulty: 'hard',
      topics: ['Proteins', 'Biochemistry']
    },
    {
      id: 5,
      passage: "Enzyme kinetics is the study of the chemical reactions that are catalyzed by enzymes.",
      question: "What is Km in enzyme kinetics?",
      options: [
        "The maximum velocity of the enzyme-catalyzed reaction",
        "The concentration of substrate at which the reaction rate is half of Vmax",
        "The turnover number of the enzyme",
        "The inhibition constant of the enzyme"
      ],
      correct: 1,
      explanation: "Km (also known as the Michaelis Constant) is the concentration at which half of the enzyme's active sites are in use and the velocity of the reaction is half of the Vmax. Km can be used to compare the affinities of different enzymes for specific substrates.",
      difficulty: 'hard',
      topics: ['Enzyme Kinetics', 'Biochemistry']
    },
    {
      id: 6,
      passage: "Enzymes play a crucial role in biological processes, catalyzing various reactions in living organisms.",
      question: "What are the four types of reversible enzyme inhibition?",
      options: [
        "Competitive, Noncompetitive, Uncompetitive, and Mixed",
        "Allosteric, Covalent, Irreversible, and Feedback",
        "Substrate, Product, Cofactor, and Coenzyme",
        "Activation, Deactivation, Phosphorylation, and Dephosphorylation"
      ],
      correct: 0,
      explanation: "The four types of reversible enzyme inhibition are Competitive, Noncompetitive, Uncompetitive, and Mixed. A helpful mnemonic to remember this is 'Cats Never Undress Men'.",
      difficulty: 'hard',
      topics: ['Enzymes', 'Biochemistry']
    },
    {
      id: 7,
      passage: "Carbohydrates are essential biomolecules with various functions in living organisms.",
      question: "What is the difference between an aldose and a ketose?",
      options: [
        "An aldose has an aldehyde group, while a ketose has a ketone group",
        "An aldose has a ketone group, while a ketose has an aldehyde group",
        "An aldose is a monosaccharide, while a ketose is a disaccharide",
        "An aldose is a pentose, while a ketose is a hexose"
      ],
      correct: 0,
      explanation: "An aldose is a carbohydrate that contains an aldehyde group as its highest priority group, while a ketose contains a ketone group as its highest priority group.",
      difficulty: 'medium',
      topics: ['Carbohydrates', 'Biochemistry']
    },
    {
      id: 8,
      passage: "Lipids are a diverse group of biomolecules with various functions in living organisms.",
      question: "What does it mean for a molecule to be amphipathic?",
      options: [
        "It has both hydrophobic and hydrophilic regions",
        "It is completely hydrophobic",
        "It is completely hydrophilic",
        "It can form covalent bonds with both water and lipids"
      ],
      correct: 0,
      explanation: "Amphipathic molecules are molecules that have both hydrophobic and hydrophilic regions. This property is important for many biological molecules, such as phospholipids in cell membranes.",
      difficulty: 'medium',
      topics: ['Lipids', 'Biochemistry']
    },
    {
      id: 9,
      passage: "Proteins are essential macromolecules with diverse functions in living organisms.",
      question: "What are the four levels of protein structure?",
      options: [
        "Primary, Secondary, Tertiary, and Quaternary",
        "Linear, Circular, Branched, and Globular",
        "Alpha, Beta, Gamma, and Delta",
        "Monomer, Dimer, Trimer, and Tetramer"
      ],
      correct: 0,
      explanation: "The four levels of protein structure are Primary (sequence of amino acids), Secondary (local folding patterns like alpha-helices and beta-sheets), Tertiary (overall 3D structure of a single polypeptide), and Quaternary (arrangement of multiple polypeptide subunits).",
      difficulty: 'medium',
      topics: ['Proteins', 'Biochemistry']
    },
    {
      id: 10,
      passage: "Amino acids are the building blocks of proteins and have various properties that affect protein structure and function.",
      question: "What is a zwitterion?",
      options: [
        "A molecule with both a positive and a negative charge simultaneously",
        "A neutral amino acid with no net charge",
        "An amino acid with only a positive charge",
        "An amino acid with only a negative charge"
      ],
      correct: 0,
      explanation: "A zwitterion is a molecule with both a positive and a negative charge simultaneously. This is seen in amino acids at neutral and body pH levels.",
      difficulty: 'hard',
      topics: ['Amino Acids', 'Biochemistry']
    },
    {
      id: 11,
      passage: "Enzymes are biological catalysts that facilitate various chemical reactions in living organisms.",
      question: "What is the Michaelis constant (Km) in enzyme kinetics?",
      options: [
        "The substrate concentration at which the reaction rate is half of Vmax",
        "The maximum velocity of the enzyme-catalyzed reaction",
        "The turnover number of the enzyme",
        "The inhibition constant of the enzyme"
      ],
      correct: 0,
      explanation: "Km (also known as the Michaelis Constant) is the concentration at which half of the enzyme's active sites are in use and the velocity of the reaction is half of the Vmax. Km can be used to compare the affinities of different enzymes for specific substrates.",
      difficulty: 'hard',
      topics: ['Enzyme Kinetics', 'Biochemistry']
    },
    {
      id: 12,
      passage: "The citric acid cycle, also known as the Krebs cycle, is a series of chemical reactions used by all aerobic organisms to release stored energy.",
      question: "Where does the citric acid cycle occur in eukaryotic cells?",
      options: [
        "Matrix of the mitochondria",
        "Cytoplasm",
        "Endoplasmic reticulum",
        "Nucleus"
      ],
      correct: 0,
      explanation: "The citric acid cycle occurs in the matrix of the mitochondria in eukaryotic cells. This location allows for the efficient coupling of the cycle with the electron transport chain.",
      difficulty: 'medium',
      topics: ['Cellular Respiration', 'Biochemistry']
    },
    {
      id: 13,
      passage: "Glycolysis is a metabolic pathway that converts glucose into pyruvate, releasing energy in the process.",
      question: "What is the net ATP yield from glycolysis?",
      options: [
        "2 ATP",
        "4 ATP",
        "36 ATP",
        "38 ATP"
      ],
      correct: 0,
      explanation: "The net ATP yield from glycolysis is 2 ATP. While 4 ATP are produced during the process, 2 ATP are consumed in the early steps, resulting in a net gain of 2 ATP.",
      difficulty: 'medium',
      topics: ['Glycolysis', 'Cellular Respiration']
    },
    {
      id: 14,
      passage: "DNA replication is the process by which DNA makes a copy of itself during cell division.",
      question: "What is the role of DNA ligase in DNA replication?",
      options: [
        "Joins together fragments of DNA, such as Okazaki fragments",
        "Unwinds the DNA double helix",
        "Adds nucleotides to the growing DNA strand",
        "Proofreads the newly synthesized DNA strand"
      ],
      correct: 0,
      explanation: "DNA ligase is an enzyme that joins together fragments of DNA, such as Okazaki fragments, during DNA replication. This is crucial for completing the synthesis of the lagging strand.",
      difficulty: 'medium',
      topics: ['DNA Replication', 'Molecular Biology']
    },
    {
      id: 15,
      passage: "Transcription is the first step of gene expression, in which a particular segment of DNA is copied into RNA by the enzyme RNA polymerase.",
      question: "What is the TATA box?",
      options: [
        "The binding site within the promoter region in eukaryotic DNA to which RNA polymerase binds",
        "A sequence of nucleotides that signals the end of transcription",
        "A region of DNA that codes for transfer RNA",
        "A protein complex that initiates translation"
      ],
      correct: 0,
      explanation: "The TATA box is the binding site within the promoter region in eukaryotic DNA to which RNA polymerase binds. It is an important regulatory element in gene transcription.",
      difficulty: 'hard',
      topics: ['Transcription', 'Molecular Biology']
    },
    {
      id: 16,
      passage: "Translation is the process by which messenger RNA (mRNA) is decoded by the ribosome to produce a specific amino acid chain, or polypeptide.",
      question: "What are the three stop codons in mRNA?",
      options: [
        "UAA, UAG, and UGA",
        "AUG, GUG, and UUG",
        "AAA, CCC, and GGG",
        "UUU, CCC, and AAA"
      ],
      correct: 0,
      explanation: "The three stop codons are UAA, UAG, and UGA. These codons signal the termination of protein synthesis during translation.",
      difficulty: 'medium',
      topics: ['Translation', 'Molecular Biology']
    },
    {
      id: 17,
      passage: "Mutations are changes in the nucleotide sequence of DNA or RNA.",
      question: "What is a frameshift mutation?",
      options: [
        "A mutation that shifts the entire reading frame of the genetic information",
        "A mutation that changes one nucleotide for another",
        "A mutation that creates a premature stop codon",
        "A mutation that does not change the amino acid sequence"
      ],
      correct: 0,
      explanation: "A frameshift mutation is a mutation that shifts the entire reading frame of the genetic information. This can be caused by insertion or deletion of nucleotides that are not in multiples of three.",
      difficulty: 'hard',
      topics: ['Mutations', 'Molecular Biology']
    },
    {
      id: 18,
      passage: "Post-transcriptional processing is a series of modifications made to the initial RNA transcript to produce mature mRNA.",
      question: "What is the function of the 5' cap in mRNA?",
      options: [
        "Protects the mRNA from degradation and is recognized by the ribosome as a binding site",
        "Signals the end of the mRNA transcript",
        "Helps in the splicing of introns",
        "Adds a poly-A tail to the mRNA"
      ],
      correct: 0,
      explanation: "The 5' cap is a methylated GTP molecule that is added during transcription. This molecule protects the mRNA from degradation in the cytoplasm and is recognized by the ribosome as a binding site.",
      difficulty: 'hard',
      topics: ['Post-transcriptional Processing', 'Molecular Biology']
    },
    {
      id: 19,
      passage: "The central dogma of molecular biology describes the flow of genetic information within a biological system.",
      question: "What is the correct order of the central dogma?",
      options: [
        "DNA → RNA → Protein",
        "RNA → DNA → Protein",
        "Protein → DNA → RNA",
        "DNA → Protein → RNA"
      ],
      correct: 0,
      explanation: "The central dogma of molecular biology states that DNA is transcribed into RNA, which is then translated into proteins. Therefore, the correct order is DNA → RNA → Protein.",
      difficulty: 'easy',
      topics: ['Central Dogma', 'Molecular Biology']
    },
    {
      id: 20,
      passage: "Cellular respiration is the process by which cells break down glucose to produce energy in the form of ATP.",
      question: "What is the primary function of the electron transport chain?",
      options: [
        "To generate a proton gradient for ATP synthesis",
        "To break down glucose into pyruvate",
        "To convert NADH back to NAD+",
        "To produce acetyl-CoA from pyruvate"
      ],
      correct: 0,
      explanation: "The primary function of the electron transport chain is to generate a proton gradient across the inner mitochondrial membrane. This gradient is then used by ATP synthase to produce ATP through oxidative phosphorylation.",
      difficulty: 'medium',
      topics: ['Cellular Respiration', 'Biochemistry']
    },
    {
      id: 21,
      passage: "Lipids are a diverse group of biomolecules that play various roles in biological systems.",
      question: "What is the primary function of cholesterol in cell membranes?",
      options: [
        "To regulate membrane fluidity",
        "To transport molecules across the membrane",
        "To provide energy for cellular processes",
        "To act as a signaling molecule"
      ],
      correct: 0,
      explanation: "Cholesterol is responsible for mediating membrane fluidity. It makes the membrane less fluid at high temperatures and more fluid at low temperatures, helping to maintain optimal membrane function across a range of temperatures.",
      difficulty: 'medium',
      topics: ['Lipids', 'Cell Biology']
    },
    {
      id: 22,
      passage: "Enzymes are biological catalysts that speed up chemical reactions in living organisms.",
      question: "What is a zymogen?",
      options: [
        "An inactive form of an enzyme that must be modified to activate it",
        "A cofactor required for enzyme function",
        "An enzyme that catalyzes its own activation",
        "A protein that inhibits enzyme activity"
      ],
      correct: 0,
      explanation: "A zymogen is an inactive form of an enzyme that must be modified to activate it. The creation of zymogens is a way to further regulate and control the activity of potentially dangerous enzymes in the body, such as digestive enzymes.",
      difficulty: 'hard',
      topics: ['Enzymes', 'Biochemistry']
    },
    {
      id: 23,
      passage: "The citric acid cycle, also known as the Krebs cycle, is a series of chemical reactions used by all aerobic organisms to release stored energy.",
      question: "What is the first step of the citric acid cycle?",
      options: [
        "Acetyl-CoA and Oxaloacetate undergo a condensation reaction to form Citrate",
        "Citrate is isomerized to Isocitrate",
        "Isocitrate is oxidized and decarboxylated to form α-Ketoglutarate",
        "α-Ketoglutarate is oxidized to Succinyl-CoA"
      ],
      correct: 0,
      explanation: "The first step of the citric acid cycle is a condensation reaction where Acetyl-CoA and Oxaloacetate combine to form Citrate, catalyzed by the enzyme Citrate Synthase.",
      difficulty: 'medium',
      topics: ['Citric Acid Cycle', 'Biochemistry']
    },
    {
      id: 24,
      passage: "Glucose transport across cell membranes is a crucial process for cellular energy metabolism.",
      question: "What is the main difference between GLUT2 and GLUT4 glucose transporters?",
      options: [
        "GLUT2 has low affinity and is found in liver and pancreas, while GLUT4 is insulin-dependent and found in muscle and fat cells",
        "GLUT2 is insulin-dependent, while GLUT4 is not",
        "GLUT2 is found in muscle cells, while GLUT4 is found in liver cells",
        "GLUT2 has high affinity for glucose, while GLUT4 has low affinity"
      ],
      correct: 0,
      explanation: "GLUT2 is a low-affinity glucose transporter found in liver and pancreatic cells, allowing glucose uptake proportional to blood glucose levels. GLUT4 is found in muscle and fat cells and is insulin-dependent, with its activity increased by insulin stimulation.",
      difficulty: 'hard',
      topics: ['Glucose Transport', 'Cell Biology']
    },
    {
      id: 25,
      passage: "DNA replication is a complex process that ensures the accurate duplication of genetic material before cell division.",
      question: "What is the role of DNA polymerase in DNA replication?",
      options: [
        "To add nucleotides to the growing DNA strand",
        "To unwind the DNA double helix",
        "To join Okazaki fragments",
        "To add the RNA primer"
      ],
      correct: 0,
      explanation: "DNA polymerase is responsible for adding nucleotides to the growing DNA strand during replication. It can only add nucleotides in the 5' to 3' direction and requires a primer to start synthesis.",
      difficulty: 'medium',
      topics: ['DNA Replication', 'Molecular Biology']
    },
    {
      id: 26,
      passage: "Protein synthesis is a complex process that involves the translation of genetic information from mRNA into a polypeptide chain.",
      question: "What is the function of tRNA in protein synthesis?",
      options: [
        "To bring amino acids to the ribosome",
        "To catalyze peptide bond formation",
        "To synthesize mRNA from DNA",
        "To degrade misfolded proteins"
      ],
      correct: 0,
      explanation: "Transfer RNA (tRNA) is responsible for bringing amino acids to the ribosome during protein synthesis. Each tRNA molecule has a specific anticodon that matches the codon on the mRNA, ensuring that the correct amino acids are added to the growing polypeptide chain.",
      difficulty: 'medium',
      topics: ['Protein Synthesis', 'Molecular Biology']
    },
    {
      id: 27,
      passage: "Gene regulation is the process by which cells control the expression of specific genes.",
      question: "What is an operon in prokaryotic gene regulation?",
      options: [
        "A cluster of genes that are transcribed as a single mRNA and regulated together",
        "A protein that represses gene transcription",
        "A small RNA molecule that regulates gene expression",
        "A DNA sequence that enhances gene transcription"
      ],
      correct: 0,
      explanation: "An operon is a cluster of genes that are transcribed as a single mRNA and regulated together. Operons are common in prokaryotes and allow for coordinated expression of genes involved in related metabolic processes.",
      difficulty: 'hard',
      topics: ['Gene Regulation', 'Molecular Biology']
    },
    {
      id: 28,
      passage: "Cellular respiration is the process by which cells break down glucose to produce energy in the form of ATP.",
      question: "What is the net ATP yield from the complete oxidation of one glucose molecule through cellular respiration?",
      options: [
        "About 30-32 ATP",
        "2 ATP",
        "38 ATP",
        "100 ATP"
      ],
      correct: 0,
      explanation: "The complete oxidation of one glucose molecule through glycolysis, the citric acid cycle, and the electron transport chain yields about 30-32 ATP molecules. This number can vary slightly depending on the efficiency of the process and the specific conditions within the cell.",
      difficulty: 'medium',
      topics: ['Cellular Respiration', 'Biochemistry']
    },
    {
      id: 29,
      passage: "Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy stored in glucose or other organic compounds.",
      question: "In which part of the chloroplast does the Calvin cycle take place?",
      options: [
        "Stroma",
        "Thylakoid membrane",
        "Intermembrane space",
        "Outer membrane"
      ],
      correct: 0,
      explanation: "The Calvin cycle, also known as the light-independent reactions or dark reactions of photosynthesis, takes place in the stroma of the chloroplast. The stroma is the fluid-filled space surrounding the thylakoid membranes.",
      difficulty: 'medium',
      topics: ['Photosynthesis', 'Plant Biology']
    },
    {
      id: 30,
      passage: "Meiosis is a type of cell division that produces gametes in sexually reproducing organisms.",
      question: "What is the main difference between mitosis and meiosis?",
      options: [
        "Meiosis produces haploid cells, while mitosis produces diploid cells",
        "Meiosis occurs only in plants, while mitosis occurs in all organisms",
        "Meiosis produces four daughter cells, while mitosis always produces two",
        "Meiosis is faster than mitosis"
      ],
      correct: 0,
      explanation: "The main difference between mitosis and meiosis is that meiosis produces haploid cells (gametes) with half the number of chromosomes as the parent cell, while mitosis produces diploid cells with the same number of chromosomes as the parent cell. Meiosis is essential for sexual reproduction and genetic diversity.",
      difficulty: 'medium',
      topics: ['Cell Division', 'Genetics']
    }
  ]
};

interface PracticeTestProps {
  onBackToDashboard: () => void;
  onUpdateStats: (stats: { xp: number; correct: number; total: number }) => void;
  addXP: (amount: number) => void;
}

export function PracticeTest({ onBackToDashboard, onUpdateStats, addXP }: PracticeTestProps) {
  const XP_RULES = {
    CORRECT_ANSWER: 10,
  };
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [answers, setAnswers] = useState({});
  const [showExplanation, setShowExplanation] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(1200); // 20 minutes
  const [isTestComplete, setIsTestComplete] = useState(false);
  const [activeTab, setActiveTab] = useState("general");

  // Timer effect for general sciences
  useEffect(() => {
    let timer;
    if (activeTab === "general" && timeRemaining > 0 && !isTestComplete) {
      timer = setInterval(() => {
        setTimeRemaining(prev => prev - 1);
      }, 1000);
    } else if (timeRemaining === 0) {
      setIsTestComplete(true);
    }
    return () => clearInterval(timer);
  }, [timeRemaining, isTestComplete, activeTab]);


  const handleAnswer = (answerIndex) => {
    setSelectedAnswer(answerIndex);
    setAnswers(prev => ({
      ...prev,
      [currentQuestion]: answerIndex
    }));
    
    // Award XP if answer is correct
    if (answerIndex === currentQuestionData.correct) {
      addXP(XP_RULES.CORRECT_ANSWER);
    }
  };

  const handleNext = () => {
    if (currentQuestion < testData.questions.length - 1) {
      setCurrentQuestion(prev => prev + 1);
      setShowExplanation(false);
      setSelectedAnswer(answers[currentQuestion + 1] ?? null);
    } else {
      setIsTestComplete(true);
    }
  };

  const handlePrevious = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(prev => prev - 1);
      setShowExplanation(false);
      setSelectedAnswer(answers[currentQuestion - 1] ?? null);
    }
  };

  const currentQuestionData = testData.questions[currentQuestion];

  const renderQuestion = () => (
    <Card className="bg-gray-800 border-0">
      <CardContent className="p-6">
        {/* Timer and Progress */}
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center space-x-4">
            <Clock className="text-blue-400" />
            <span className="text-xl font-bold">
              {Math.floor(timeRemaining / 60)}:{String(timeRemaining % 60).padStart(2, '0')}
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-gray-400">Question</span>
            <span className="font-bold">{currentQuestion + 1}</span>
            <span className="text-gray-400">of {testData.questions.length}</span>
          </div>
        </div>

        {/* Back to Dashboard */}
        <Button
          variant="ghost"
          className="mb-4"
          onClick={onBackToDashboard}
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Dashboard
        </Button>

        {/* Passage */}
        <div className="bg-gray-700 rounded-lg p-4 mb-6">
          <div className="prose prose-invert max-w-none">
            {currentQuestionData.passage}
          </div>
        </div>

        {/* Question */}
        <div className="mb-6">
          <h3 className="text-lg font-medium mb-4">
            {currentQuestionData.question}
          </h3>

          {/* Answer Options */}
          <div className="space-y-3">
            {currentQuestionData.options.map((option, index) => (
              <button
                key={index}
                onClick={() => handleAnswer(index)}
                className={`w-full text-left p-4 rounded-lg transition-colors ${
                  selectedAnswer === index 
                    ? showExplanation
                      ? index === currentQuestionData.correct
                        ? 'bg-green-500/20 text-green-400'
                        : 'bg-red-500/20 text-red-400'
                      : 'bg-blue-500 text-white'
                    : 'bg-gray-700 hover:bg-gray-600'
                }`}
              >
                <div className="flex items-center">
                  <span className="mr-3">{String.fromCharCode(65 + index)}.</span>
                  <span>{option}</span>
                  {showExplanation && index === currentQuestionData.correct && (
                    <CheckCircle className="ml-auto text-green-400" />
                  )}
                  {showExplanation && selectedAnswer === index && index !== currentQuestionData.correct && (
                    <AlertCircle className="ml-auto text-red-400" />
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-between items-center">
          <Button
            onClick={handlePrevious}
            disabled={currentQuestion === 0}
            variant="outline"
            className="flex items-center space-x-2"
          >
            <ChevronLeft className="h-4 w-4" />
            <span>Previous</span>
          </Button>

          <div className="flex space-x-3">
            <Button
              onClick={() => setShowExplanation(!showExplanation)}
              variant="outline"
              className="flex items-center space-x-2"
            >
              {showExplanation ? 'Hide' : 'Show'} Explanation
            </Button>
            <Button
              onClick={handleNext}
              className="flex items-center space-x-2"
              disabled={selectedAnswer === null}
            >
              <span>{currentQuestion === testData.questions.length - 1 ? 'Finish' : 'Next'}</span>
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Explanation */}
        {showExplanation && (
          <div className="mt-6 bg-gray-700 rounded-lg p-4">
            <div className="font-medium mb-2">Explanation:</div>
            <div className="text-gray-300 whitespace-pre-line">
              {currentQuestionData.explanation}
            </div>
            <div className="mt-2 text-sm">
              <span className="text-gray-400">Topics: </span>
              {currentQuestionData.topics.join(', ')}
            </div>
            <div className="mt-1 text-sm">
              <span className="text-gray-400">Difficulty: </span>
              <span className={
                currentQuestionData.difficulty === 'hard' ? 'text-red-400' :
                currentQuestionData.difficulty === 'medium' ? 'text-yellow-400' :
                'text-green-400'
              }>
                {currentQuestionData.difficulty}
              </span>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );

  const renderStatistics = () => (
    <Card className="mt-6 bg-gray-800 border-0">
      <CardHeader>
        <CardTitle>Your Performance</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-gray-400">Accuracy</div>
            <div className="text-2xl font-bold mt-1">{Math.round(performance.accuracy)}%</div>
            <Progress 
              value={performance.accuracy}
              className="h-2 bg-gray-600 mt-2"
            />
          </div>

          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-gray-400">Questions Completed</div>
            <div className="text-2xl font-bold mt-1">
              {performance.questionsAnswered}/{testData.questions.length}
            </div>
            <Progress 
              value={(performance.questionsAnswered/testData.questions.length) * 100}
              className="h-2 bg-gray-600 mt-2"
            />
          </div>

          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-gray-400">Average Time per Question</div>
            <div className="text-2xl font-bold mt-1">{performance.averageTime}s</div>
            <div className="text-sm text-gray-400 mt-1">Target: 60s</div>
          </div>

          <div className="bg-gray-700 rounded-lg p-4">
            <div className="text-gray-400">Your Percentile</div>
            <div className="text-2xl font-bold mt-1">{performance.comparison.yourPercentile}th</div>
            <div className="text-sm text-gray-400 mt-1">Among all test takers</div>
          </div>
        </div>

        {/* Comparative Analytics */}
        <div className="bg-gray-700 rounded-lg p-4">
          <h4 className="font-medium mb-4">Performance Comparison</h4>
          <div className="spacey-4">
            {[
              { label: 'Top Performers', value: performance.comparison.topPerformers },
              { label: 'Your Score', value: performance.accuracy },
              { label: 'Average Score', value: performance.comparison.average }
            ].map(metric => (
              <div key={metric.label} className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span>{metric.label}</span>
                  <span>{Math.round(metric.value)}%</span>
                </div>
                <Progress
                  value={metric.value}
                  className={`h-2 ${
                    metric.label === 'Your Score' ? 'bg-blue-500' :
                    metric.label === 'Top Performers' ? 'bg-green-500' :
                    'bg-yellow-500'
                  }`}
                />
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  );

  // Calculate performance metrics
  const performance = {
    questionsAnswered: Object.keys(answers).length,
    correctAnswers: Object.entries(answers).filter(
      ([qIndex, answer]) => answer === testData.questions[parseInt(qIndex)].correct
    ).length,
    accuracy: Object.entries(answers).filter(
      ([qIndex, answer]) => answer === testData.questions[parseInt(qIndex)].correct
    ).length / Object.keys(answers).length * 100 || 0,
    averageTime: Math.round((1200 - timeRemaining) / Math.max(1, Object.keys(answers).length)),
    comparison: {
      topPerformers: 88,
      average: 72,
      yourPercentile: 85
    }
  };

  const handleStatsUpdate = ({ xp, correct, total }) => {
    console.log('Practice Test - Stats Update:', { xp, correct, total });
    addXP(xp); // Use the XP directly from CARS practice
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-4xl mx-auto">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="general">General Sciences</TabsTrigger>
            <TabsTrigger value="cars">CARS</TabsTrigger>
            <TabsTrigger value="time-attack">Time Attack</TabsTrigger>
            <TabsTrigger value="med-tutor">MedTutor</TabsTrigger>
          </TabsList>
          <TabsContent value="general">
            {renderQuestion()}
            {isTestComplete ? renderStatistics() : null}
          </TabsContent>
          <TabsContent value="cars">
            <CARSPractice onExitToMenu={() => setActiveTab('general')} onUpdateStats={handleStatsUpdate} />
          </TabsContent>
          <TabsContent value="time-attack">
            <TimeAttackMode onUpdateStats={handleStatsUpdate} addXP={addXP} />
          </TabsContent>
          <TabsContent value="med-tutor">
            <MedTutor addXP={addXP} />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

