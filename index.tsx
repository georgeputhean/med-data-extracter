import React, { useState, useRef, useEffect, FormEvent } from "react";
import { createRoot } from "react-dom/client";
import { GoogleGenAI, Type, FunctionDeclaration, LiveServerMessage, Modality } from "@google/genai";

// --- Types ---
type AppMode = 'intake' | 'sales';

interface PatientData {
  fullName: string;
  dob: string;
  insuranceProvider: string;
  policyNumber: string;
  planType: string;
  copay: string;
  deductible: string;
  notes: string;
}

interface SalesData {
  physicianName: string;
  specialty: string;
  clinicName: string;
  productDiscussed: string;
  samplesLeft: string;
  nextVisitDate: string;
  notes: string;
}

interface Message {
  role: "user" | "ai";
  text: string;
}

// --- Helpers for Audio ---
function createBlob(data: Float32Array) {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  let binary = "";
  const len = int16.buffer.byteLength;
  const bytes = new Uint8Array(int16.buffer);
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return {
    data: btoa(binary),
    mimeType: "audio/pcm;rate=16000",
  };
}

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

// --- Initial States ---
const initialPatientData: PatientData = {
  fullName: "",
  dob: "",
  insuranceProvider: "",
  policyNumber: "",
  planType: "",
  copay: "",
  deductible: "",
  notes: "",
};

const initialSalesData: SalesData = {
  physicianName: "",
  specialty: "",
  clinicName: "",
  productDiscussed: "",
  samplesLeft: "",
  nextVisitDate: "",
  notes: "",
};

// --- Gemini Configuration ---

// 1. Patient Intake Config
const updatePatientTool: FunctionDeclaration = {
  name: "updatePatientRecord",
  description: "Updates the patient information record with extracted details from the conversation.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      fullName: { type: Type.STRING, description: "Patient's full legal name" },
      dob: { type: Type.STRING, description: "Date of birth or Age" },
      insuranceProvider: { type: Type.STRING, description: "Name of the insurance company (e.g. Aetna, BCBS)" },
      policyNumber: { type: Type.STRING, description: "Member ID or Policy Number" },
      planType: { type: Type.STRING, description: "Specific plan name (e.g. Gold PPO, HMO)" },
      copay: { type: Type.STRING, description: "Copay amount (e.g. $20, $50)" },
      deductible: { type: Type.STRING, description: "Deductible amount if mentioned" },
      notes: { type: Type.STRING, description: "Medical notes, symptoms, or chief complaint" },
    },
    required: [] 
  },
};

const PATIENT_SYSTEM_INSTRUCTION = `You are an efficient and helpful Medical Intake Assistant. 
Your goal is to extract patient information from the user (who is a receptionist or nurse) to populate the patient record.
1. ALWAYS use the 'updatePatientRecord' tool when new information is provided.
2. If the user provides multiple details, extract all of them in one tool call.
3. Be conversational but concise. Confirm what you have recorded.
4. If critical info (Name, Insurance, Copay) is missing, politely ask for it, but do not be repetitive.
5. If the user corrects information, overwrite the previous value using the tool.`;

// 2. Sales Rep Config
const updateSalesTool: FunctionDeclaration = {
  name: "updateSalesRecord",
  description: "Updates the sales visit report with extracted details from the conversation.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      physicianName: { type: Type.STRING, description: "Name of the physician/HCP visited" },
      specialty: { type: Type.STRING, description: "Primary specialty of the physician" },
      clinicName: { type: Type.STRING, description: "Name of the clinic, hospital, or practice" },
      productDiscussed: { type: Type.STRING, description: "Key product(s) discussed during the visit" },
      samplesLeft: { type: Type.STRING, description: "Details on samples or materials left (e.g. '5 boxes of X')" },
      nextVisitDate: { type: Type.STRING, description: "Planned date or timeframe for the follow-up visit" },
      notes: { type: Type.STRING, description: "Key insights, objections, questions asked, or general notes" },
    },
    required: []
  },
};

const SALES_SYSTEM_INSTRUCTION = `You are an efficient Sales Assistant for a pharmaceutical representative.
Your goal is to help the rep log their visit with a Healthcare Professional (HCP).
1. ALWAYS use the 'updateSalesRecord' tool when information about the visit is provided.
2. Extract the Physician's name, specialty, and clinic details.
3. Record products discussed and any samples left.
4. Note any follow-up dates or important qualitative notes/objections.
5. Be concise and professional.`;

function App() {
  const [activeTab, setActiveTab] = useState<AppMode>('intake');
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  
  // Data States
  const [patientData, setPatientData] = useState<PatientData>(initialPatientData);
  const [salesData, setSalesData] = useState<SalesData>(initialSalesData);
  
  const [lastUpdatedFields, setLastUpdatedFields] = useState<Set<string>>(new Set());
  const [isLiveConnected, setIsLiveConnected] = useState(false);

  // Refs
  const chatSessionRef = useRef<any>(null);
  const liveSessionRef = useRef<any>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const audioSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  const nextStartTimeRef = useRef<number>(0);

  // --- Configuration Management ---
  const getConfig = (mode: AppMode) => {
    if (mode === 'sales') {
      return {
        instruction: SALES_SYSTEM_INSTRUCTION,
        tools: [{ functionDeclarations: [updateSalesTool] }],
        initialMessage: "Ready to log your sales visit. Who did you see today?"
      };
    } else {
      return {
        instruction: PATIENT_SYSTEM_INSTRUCTION,
        tools: [{ functionDeclarations: [updatePatientTool] }],
        initialMessage: "Hello. I'm ready to assist with patient intake. Please provide the patient details."
      };
    }
  };

  // Initialize Chat Session when Tab Changes
  useEffect(() => {
    // 1. Disconnect Live if active to prevent state mismatch
    if (isLiveConnected) {
      disconnectLive();
    }

    // 2. Reset Messages
    const config = getConfig(activeTab);
    setMessages([{ role: "ai", text: config.initialMessage }]);
    
    // 3. Initialize Text Chat Client
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    chatSessionRef.current = ai.chats.create({
      model: "gemini-3-flash-preview",
      config: {
        systemInstruction: config.instruction,
        tools: config.tools,
      },
    });

  }, [activeTab]); // Dependency on activeTab ensures re-init

  // Clean up Audio on unmount
  useEffect(() => {
    return () => {
      disconnectLive();
    };
  }, []);

  // --- Logic for Text Chat ---

  const handleSendMessage = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: userMsg }]);
    setIsLoading(true);

    try {
      if (!chatSessionRef.current) return;

      let response = await chatSessionRef.current.sendMessage({ message: userMsg });
      
      const functionCalls = response.candidates?.[0]?.content?.parts?.filter((p: any) => p.functionCall)?.map((p: any) => p.functionCall);

      if (functionCalls && functionCalls.length > 0) {
        const toolResponses = [];
        const newUpdates = new Set<string>();

        for (const call of functionCalls) {
          if (call.name === "updatePatientRecord" && activeTab === 'intake') {
             const args = call.args as Partial<PatientData>;
             setPatientData(prev => {
               const updated = { ...prev };
               Object.keys(args).forEach(key => {
                 const k = key as keyof PatientData;
                 if (args[k]) {
                   updated[k] = args[k] as string;
                   newUpdates.add(k);
                 }
               });
               return updated;
             });
             toolResponses.push({ name: call.name, response: { result: "Patient record updated." }, id: call.id });
          } 
          else if (call.name === "updateSalesRecord" && activeTab === 'sales') {
             const args = call.args as Partial<SalesData>;
             setSalesData(prev => {
               const updated = { ...prev };
               Object.keys(args).forEach(key => {
                 const k = key as keyof SalesData;
                 if (args[k]) {
                   updated[k] = args[k] as string;
                   newUpdates.add(k);
                 }
               });
               return updated;
             });
             toolResponses.push({ name: call.name, response: { result: "Sales record updated." }, id: call.id });
          }
        }
        
        setLastUpdatedFields(newUpdates);
        setTimeout(() => setLastUpdatedFields(new Set()), 2000);

        if (toolResponses.length > 0) {
           response = await chatSessionRef.current.sendMessage({
             message: toolResponses.map(tr => ({
               functionResponse: {
                 name: tr.name,
                 response: tr.response,
                 id: tr.id
               }
             }))
           });
        }
      }

      const aiText = response.text;
      if (aiText) {
        setMessages((prev) => [...prev, { role: "ai", text: aiText }]);
      }

    } catch (error) {
      console.error("Error communicating with AI:", error);
      setMessages((prev) => [...prev, { role: "ai", text: "Sorry, I encountered an error processing that." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(e as any);
    }
  };

  // --- Logic for Live API (Audio) ---

  const connectLive = async () => {
    if (isLiveConnected) return;

    try {
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const config = getConfig(activeTab);

      // Setup Audio Contexts
      inputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      outputAudioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const sessionPromise = ai.live.connect({
        model: "gemini-2.5-flash-native-audio-preview-12-2025",
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: config.instruction,
          tools: config.tools,
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }
          }
        },
        callbacks: {
          onopen: () => {
            console.log("Live Session Connected");
            setIsLiveConnected(true);
            
            const source = inputAudioContextRef.current!.createMediaStreamSource(stream);
            const scriptProcessor = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
            
            scriptProcessor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const pcmBlob = createBlob(inputData);
              sessionPromise.then(session => session.sendRealtimeInput({ media: pcmBlob }));
            };
            
            source.connect(scriptProcessor);
            scriptProcessor.connect(inputAudioContextRef.current!.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
             // 1. Handle Audio Output
             const base64Audio = message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
             if (base64Audio) {
               const ctx = outputAudioContextRef.current;
               if (ctx) {
                 nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
                 const audioBuffer = await decodeAudioData(decode(base64Audio), ctx, 24000, 1);
                 const source = ctx.createBufferSource();
                 source.buffer = audioBuffer;
                 source.connect(ctx.destination);
                 source.start(nextStartTimeRef.current);
                 nextStartTimeRef.current += audioBuffer.duration;
                 audioSourcesRef.current.add(source);
                 source.onended = () => audioSourcesRef.current.delete(source);
               }
             }

             // 2. Handle Tool Calls
             if (message.toolCall) {
               for (const fc of message.toolCall.functionCalls) {
                 const newUpdates = new Set<string>();
                 
                 if (fc.name === "updatePatientRecord" && activeTab === 'intake') {
                   const args = fc.args as Partial<PatientData>;
                   setPatientData(prev => {
                     const updated = { ...prev };
                     Object.keys(args).forEach(key => {
                       const k = key as keyof PatientData;
                       if (args[k]) { updated[k] = args[k] as string; newUpdates.add(k); }
                     });
                     return updated;
                   });
                 } else if (fc.name === "updateSalesRecord" && activeTab === 'sales') {
                   const args = fc.args as Partial<SalesData>;
                   setSalesData(prev => {
                     const updated = { ...prev };
                     Object.keys(args).forEach(key => {
                       const k = key as keyof SalesData;
                       if (args[k]) { updated[k] = args[k] as string; newUpdates.add(k); }
                     });
                     return updated;
                   });
                 }

                 setLastUpdatedFields(newUpdates);
                 setTimeout(() => setLastUpdatedFields(new Set()), 2000);

                 sessionPromise.then(session => {
                   session.sendToolResponse({
                     functionResponses: [{ id: fc.id, name: fc.name, response: { result: "Updated" } }]
                   });
                 });
               }
             }
          },
          onclose: () => {
            console.log("Live Session Closed");
            setIsLiveConnected(false);
          },
          onerror: (err) => {
            console.error("Live Session Error:", err);
            setIsLiveConnected(false);
          }
        }
      });
      
      liveSessionRef.current = sessionPromise;

    } catch (e) {
      console.error("Failed to connect live:", e);
      setIsLiveConnected(false);
    }
  };

  const disconnectLive = async () => {
    if (liveSessionRef.current) {
      try {
        const session = await liveSessionRef.current;
        session.close();
      } catch (e) { console.error(e); }
      liveSessionRef.current = null;
    }
    
    if (inputAudioContextRef.current) {
      inputAudioContextRef.current.close();
      inputAudioContextRef.current = null;
    }
    if (outputAudioContextRef.current) {
      outputAudioContextRef.current.close();
      outputAudioContextRef.current = null;
    }
    setIsLiveConnected(false);
  };

  const toggleLive = () => {
    if (isLiveConnected) {
      disconnectLive();
    } else {
      connectLive();
    }
  };


  // --- UI Components ---

  const Field = ({ label, fieldKey, value }: { label: string, fieldKey: string, value: string }) => {
    const isUpdated = lastUpdatedFields.has(fieldKey);
    return (
      <div className="field-group">
        <label className="label">{label}</label>
        <div className={`value ${value ? '' : 'empty'} ${isUpdated ? 'updated' : ''}`}>
          {value || "---"}
        </div>
      </div>
    );
  };

  const TabButton = ({ mode, label }: { mode: AppMode, label: string }) => (
    <button
      onClick={() => setActiveTab(mode)}
      style={{
        padding: '0.75rem 1.5rem',
        borderBottom: activeTab === mode ? '3px solid var(--primary)' : '3px solid transparent',
        background: 'transparent',
        borderTop: 'none', borderLeft: 'none', borderRight: 'none',
        color: activeTab === mode ? 'var(--primary)' : 'var(--text-muted)',
        fontWeight: activeTab === mode ? 700 : 500,
        cursor: 'pointer',
        fontSize: '1rem',
        transition: 'all 0.2s'
      }}
    >
      {label}
    </button>
  );

  return (
    <div className="h-full flex-col">
      <header className="header" style={{ paddingBottom: 0, flexDirection: 'column', alignItems: 'flex-start', gap: '1rem' }}>
        <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="brand">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 6v12M6 12h12" />
              <rect width="18" height="18" x="3" y="3" rx="2" />
            </svg>
            MedExtract AI
          </div>
          <button 
             onClick={() => {
               if (activeTab === 'intake') setPatientData(initialPatientData);
               if (activeTab === 'sales') setSalesData(initialSalesData);
               setMessages([{ role: "ai", text: "Record cleared. Ready." }]);
             }}
             style={{ background: 'transparent', border: '1px solid #cbd5e1', padding: '0.4rem 0.8rem', borderRadius: '0.5rem', cursor: 'pointer', color: '#64748b', fontSize: '0.9rem' }}
           >
             Clear Form
           </button>
        </div>
        
        <div style={{ display: 'flex', gap: '1rem' }}>
          <TabButton mode="intake" label="Patient Intake" />
          <TabButton mode="sales" label="Sales Visit Rep" />
        </div>
      </header>

      <div className="main-container">
        {/* Chat Section */}
        <div className="chat-section">
          <div className="messages-area">
            {messages.map((msg, i) => (
              <div key={i} className={`message-bubble ${msg.role}`}>
                {msg.text}
              </div>
            ))}
            {isLoading && (
              <div className="message-bubble ai" style={{ opacity: 0.7 }}>
                <span className="typing-dot">...</span>
              </div>
            )}
            <div ref={(el) => el?.scrollIntoView({ behavior: 'smooth' })} />
          </div>
          
          <form className="input-area" onSubmit={handleSendMessage}>
            <button 
               type="button" 
               className={`mic-btn ${isLiveConnected ? 'active' : ''}`}
               onClick={toggleLive}
               title={isLiveConnected ? "Stop Voice Session" : "Start Voice Session"}
            >
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                {isLiveConnected ? (
                  <>
                    <path d="M9 9v3a3 3 0 0 0 6 0V9" />
                    <path d="M12 19v3" />
                    <path d="M8 22h8" />
                    <path d="M19 12a7 7 0 0 1-14 0" />
                  </>
                ) : (
                  <>
                    <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                    <line x1="12" x2="12" y1="19" y2="22" />
                  </>
                )}
              </svg>
            </button>

            <textarea
              className="chat-input"
              placeholder={isLiveConnected ? "Listening..." : "Type details here..."}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading} 
              rows={1}
            />
            <button className="send-btn" type="submit" disabled={isLoading || !input.trim()}>
              Send
            </button>
          </form>
        </div>

        {/* Data Section */}
        <div className="data-section">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">
                {activeTab === 'intake' ? 'Patient Record' : 'Sales Visit Report'}
              </h2>
              <p className="card-subtitle">Real-time data extraction</p>
            </div>
            <div className="data-grid">
              
              {activeTab === 'intake' ? (
                // PATIENT FIELDS
                <>
                  <Field label="Full Name" fieldKey="fullName" value={patientData.fullName} />
                  <Field label="DOB / Age" fieldKey="dob" value={patientData.dob} />
                  <div style={{ height: '1px', background: 'var(--border)', margin: '0.5rem 0' }} />
                  <Field label="Insurance Provider" fieldKey="insuranceProvider" value={patientData.insuranceProvider} />
                  <Field label="Plan Type" fieldKey="planType" value={patientData.planType} />
                  <Field label="Policy / Member ID" fieldKey="policyNumber" value={patientData.policyNumber} />
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <Field label="Copay" fieldKey="copay" value={patientData.copay} />
                    <Field label="Deductible" fieldKey="deductible" value={patientData.deductible} />
                  </div>
                  <div style={{ height: '1px', background: 'var(--border)', margin: '0.5rem 0' }} />
                  <Field label="Notes / Symptoms" fieldKey="notes" value={patientData.notes} />
                </>
              ) : (
                // SALES FIELDS
                <>
                  <Field label="Physician / HCP Name" fieldKey="physicianName" value={salesData.physicianName} />
                  <Field label="Specialty" fieldKey="specialty" value={salesData.specialty} />
                  <Field label="Clinic / Hospital" fieldKey="clinicName" value={salesData.clinicName} />
                  <div style={{ height: '1px', background: 'var(--border)', margin: '0.5rem 0' }} />
                  <Field label="Product Discussed" fieldKey="productDiscussed" value={salesData.productDiscussed} />
                  <Field label="Samples Left" fieldKey="samplesLeft" value={salesData.samplesLeft} />
                  <div style={{ height: '1px', background: 'var(--border)', margin: '0.5rem 0' }} />
                  <Field label="Next Visit Date" fieldKey="nextVisitDate" value={salesData.nextVisitDate} />
                  <Field label="Notes / Objections" fieldKey="notes" value={salesData.notes} />
                </>
              )}

            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const root = createRoot(document.getElementById("root")!);
root.render(<App />);