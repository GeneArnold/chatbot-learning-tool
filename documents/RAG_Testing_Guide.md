# RAG Testing Guide: Systematic Exploration Scripts

## Overview
This guide provides a structured approach to testing and understanding your RAG system using the comprehensive interface we've built. The guide is divided into **Core Tests** for demonstrations and **Extra Testing** for extended exploration and experimentation.

---

## 🎯 **Testing Philosophy**

Each test is designed to:
- **Validate** a specific aspect of the RAG system
- **Teach** you something about how RAG works
- **Reveal** optimization opportunities
- **Build** your intuition about parameter effects
- **Demonstrate** advanced features like query expansion

---

## 📋 **Pre-Test Setup**

Before starting any test series:
1. **Clear all data** using the "🗑️ Clear All Data" button
2. **Set baseline parameters** using "🎯 Precise Mode" preset
3. **Have test documents ready** - We have 4 excellent documents available:
   - `empire_state_knowledge.md` - 1930s skyscraper
   - `golden_gate_bridge.md` - 1930s bridge engineering
   - `hoover_dam.md` - 1930s concrete dam
   - `international_space_station.md` - Modern space engineering
4. **Enable Query Expansion** (default ON) in Parameters tab

---

# 🏆 **CORE TESTS** (For Demonstrations)

## 🧪 **TEST SERIES 1: BASIC RAG VALIDATION**

### Test 1.1: Baseline vs RAG Comparison
**Objective:** Prove that RAG actually improves answers with relevant context

**What You'll Learn:** Whether RAG provides measurable improvement over general knowledge

**Steps:**
1. **Ask without documents:**
   - Go to "💬 Ask Questions" tab
   - Ask: "How tall is the Golden Gate Bridge?"
   - Note the answer type (should be "General")
   - Save the response

2. **Upload Golden Gate Bridge document:**
   - Go to "📁 Documents" tab
   - Upload `golden_gate_bridge.md` from demo_context_files
   - Process documents with default settings

3. **Ask the same question:**
   - Return to "💬 Ask Questions" tab
   - Ask the exact same question
   - Note the answer type (should be "RAG")
   - Compare specificity and accuracy

**Expected Results:**
- General answer: Vague or potentially incorrect
- RAG answer: Specific "746 feet (227 meters) above water"

**What This Proves:** RAG retrieval provides more accurate, contextual answers

---

### Test 1.2: Query Expansion Effectiveness
**Objective:** Demonstrate how query expansion improves natural language question handling

**What You'll Learn:** Why different question phrasings can yield different results

**Steps:**
1. **Test with query expansion ON (default):**
   - Ensure "🔍 Enable Query Expansion" is checked in Parameters
   - Ask: "How tall is the tallest tower of the Golden Gate Bridge?"
   - Go to "🔍 RAG Details" to see retrieved chunks
   - Note the answer quality

2. **Test with query expansion OFF:**
   - Uncheck "🔍 Enable Query Expansion" in Parameters
   - Ask the same question
   - Compare retrieval results and answer quality

3. **Test keyword-style query:**
   - Ask: "Golden Gate Bridge tower height"
   - Compare with the natural language version

**Expected Results:**
- Query expansion ON: Successfully finds tower height information
- Query expansion OFF: May struggle with natural language phrasing
- Keyword query: Works well regardless of expansion setting

**What This Proves:** Query expansion bridges the gap between natural language and document structure

---

## 🔬 **TEST SERIES 2: PARAMETER OPTIMIZATION**

### Test 2.1: The Golden Gate Bridge Solution
**Objective:** Demonstrate the parameter optimization process using a real-world example

**What You'll Learn:** How systematic parameter adjustment solves retrieval problems

**Steps:**
1. **Start with default parameters:**
   - Chunk Size: 500, Overlap: 100, Results: 3
   - Ask: "How tall is the tallest tower of the Golden Gate Bridge?"
   - Note if answer is incomplete or says "insufficient information"

2. **Apply the optimized solution:**
   - Change to: Chunk Size: 700, Overlap: 125, Results: 8
   - Reprocess documents
   - Ask the same question
   - Note the dramatic improvement

3. **Test the individual parameter effects:**
   - Try just increasing chunk size to 700 (keep others default)
   - Try just increasing results to 8 (keep others default)
   - See which parameter had the biggest impact

**Expected Results:**
- Default parameters: May fail to find the information
- Optimized parameters: Clear, accurate answer with proper context
- Individual testing: Reveals which parameters matter most

**What This Proves:** Systematic parameter optimization can solve challenging retrieval problems

---

## 🎨 **TEST SERIES 3: MULTI-DOCUMENT TESTING**

### Test 3.1: Cross-Document Comparison
**Objective:** Test the system's ability to compare information across multiple documents

**What You'll Learn:** How RAG handles multi-document queries

**Steps:**
1. **Upload multiple documents:**
   - Upload both `empire_state_knowledge.md` and `golden_gate_bridge.md`
   - Process with optimized parameters (700/125/8)

2. **Test comparative queries:**
   - Ask: "Which is taller: the Empire State Building or the Golden Gate Bridge towers?"
   - Ask: "Compare the construction timelines of these two landmarks"
   - Ask: "What materials were used in both structures?"

3. **Test document-specific queries:**
   - Ask: "What is unique about the Golden Gate Bridge's color?"
   - Verify it doesn't confuse information between documents

**Expected Results:**
- Comparative queries: Should find relevant information from both documents
- Specific queries: Should maintain document attribution
- No information bleeding between unrelated topics

**What This Proves:** The system can handle multiple documents while maintaining accuracy

---

# 🚀 **EXTRA TESTING** (Extended Exploration)

## 📊 **CATEGORY A: ADVANCED PARAMETER STUDIES**

### A.1: Chunk Size Deep Dive
**Objective:** Understand optimal chunk sizes for different content types

**Test Scenarios:**
1. **Technical Specifications** (ISS document):
   - Test 200, 500, 800, 1200 character chunks
   - Ask: "What are the ISS solar array specifications?"
   - Find optimal size for dense technical content

2. **Historical Narrative** (Empire State Building):
   - Test same chunk sizes
   - Ask: "Describe the Empire State Building's construction story"
   - Compare narrative flow quality

3. **Mixed Content** (Hoover Dam):
   - Test with construction details and statistics
   - Ask: "How was the Hoover Dam's concrete cooling system designed?"
   - Find balance between detail and context

### A.2: Overlap Optimization Study
**Objective:** Find the sweet spot for chunk overlap

**Test Matrix:**
- Chunk Size: 600 (fixed)
- Overlap: 0, 50, 100, 150, 200, 250
- Query: "Describe the Golden Gate Bridge construction challenges"
- Measure: Answer completeness and flow

**Analysis Points:**
- At what overlap do answers become smooth?
- When does overlap become redundant?
- How does overlap affect processing time?

### A.3: Retrieval Count Impact Analysis
**Objective:** Determine optimal number of chunks for different query types

**Query Categories:**
1. **Simple Factual**: "When was the ISS first occupied?" (test 1-10 results)
2. **Complex Analytical**: "Compare engineering challenges across all four structures" (test 5-20 results)
3. **Synthesis**: "How do these structures represent different eras of engineering?" (test 10-30 results)

---

## 🧠 **CATEGORY B: QUERY COMPLEXITY STUDIES**

### B.1: Natural Language vs Keyword Optimization
**Objective:** Master query phrasing for best results

**Natural Language Variations:**
- "How tall is the tallest tower of the Golden Gate Bridge?"
- "What's the height of the Golden Gate Bridge's highest point?"
- "How high do the Golden Gate Bridge towers reach?"

**Keyword Variations:**
- "Golden Gate Bridge tower height"
- "bridge tower 746 feet"
- "tower height above water"

**Test Process:**
1. Test each variation with query expansion ON/OFF
2. Measure retrieval quality and answer accuracy
3. Document which phrasings work best

### B.2: Multi-Part Question Handling
**Objective:** Test complex queries requiring multiple pieces of information

**Progressive Complexity:**
1. **Two-part**: "What is the height and construction date of the Empire State Building?"
2. **Three-part**: "Compare the height, construction time, and cost of the Empire State Building and Hoover Dam"
3. **Synthesis**: "How do the construction challenges of the 1930s projects compare to modern ISS assembly?"

### B.3: Hypothetical and Inference Testing
**Objective:** Understand RAG limitations with questions requiring inference

**Test Questions:**
- "What would happen if the Golden Gate Bridge were built today?"
- "How might the ISS design influence future space stations?"
- "What if the Hoover Dam had been built with modern materials?"

**Expected Results:**
- RAG should acknowledge when information isn't in the context
- System should not hallucinate answers beyond provided information

---

## 🔬 **CATEGORY C: TECHNICAL SYSTEM ANALYSIS**

### C.1: Vector Similarity Deep Dive
**Objective:** Understand the relationship between similarity scores and content relevance

**Analysis Steps:**
1. **High Similarity Analysis** (scores < 0.3):
   - Find chunks with excellent similarity scores
   - Examine their actual content relevance
   - Look for false positives

2. **Medium Similarity Analysis** (scores 0.3-0.6):
   - Assess whether these chunks provide useful context
   - Identify borderline cases

3. **Low Similarity Analysis** (scores > 0.6):
   - Determine if these are noise or still valuable
   - Test if increasing retrieval count helps or hurts

### C.2: Query Expansion Mechanism Study
**Objective:** Understand exactly how query expansion works

**Detailed Analysis:**
1. **Enable query expansion** and ask: "How tall is the tallest tower of the Golden Gate Bridge?"
2. **Manually test the expanded queries** (with expansion OFF):
   - "Golden Gate Bridge tower height"
   - "tower height 746 feet"
   - "bridge tower dimensions"
3. **Compare retrieval results** for each variation
4. **Document which expansions are most effective**

### C.3: Cross-Document Retrieval Patterns
**Objective:** Understand how the system handles information from multiple sources

**Test Scenarios:**
1. **Upload all 4 documents** (Empire State, Golden Gate, Hoover Dam, ISS)
2. **Test queries that could match multiple documents**:
   - "tallest structure" (could match any)
   - "1930s construction" (matches 3 documents)
   - "engineering marvel" (matches all)
3. **Analyze which documents are retrieved** and why

---

## 🎯 **CATEGORY D: DOMAIN-SPECIFIC TESTING**

### D.1: Era-Based Comparisons
**Objective:** Test the system's ability to handle temporal relationships

**1930s Engineering Projects:**
- Empire State Building (1930-1931)
- Golden Gate Bridge (1933-1937)  
- Hoover Dam (1931-1936)

**Test Queries:**
- "What were the common challenges of 1930s mega-projects?"
- "How did Great Depression affect these construction projects?"
- "Compare 1930s engineering techniques to modern ISS assembly"

### D.2: Material Science Comparisons
**Objective:** Test technical domain knowledge across documents

**Material Categories:**
- **Steel**: Empire State Building, Golden Gate Bridge
- **Concrete**: Hoover Dam, Empire State Building (mixed)
- **Advanced Materials**: ISS (aluminum, composites)

**Test Queries:**
- "Compare steel usage across these structures"
- "How do concrete and steel construction methods differ?"
- "What advanced materials are used in space construction?"

### D.3: Scale and Dimension Analysis
**Objective:** Test numerical reasoning and comparison capabilities

**Dimension Comparisons:**
- **Height**: Empire State (1,454 ft) vs Golden Gate towers (746 ft) vs Hoover Dam (726 ft)
- **Length**: Golden Gate Bridge (8,980 ft) vs ISS (357 ft)
- **Weight**: Various massive structures

**Test Queries:**
- "Rank these structures by height"
- "Which structure required the most material?"
- "Compare the scales of terrestrial vs space engineering"

---

## 🌟 **CATEGORY E: CREATIVE AND EDGE CASE TESTING**

### E.1: Ambiguous Query Resolution
**Objective:** Test how the system handles unclear or ambiguous questions

**Ambiguous Queries:**
- "How big is it?" (without specifying which structure)
- "When was it built?" (could refer to any structure)
- "What makes it special?" (very general)

**Expected Behavior:**
- System should ask for clarification or provide context-appropriate answers
- Should not make assumptions about which structure is referenced

### E.2: Negative and Limitation Testing
**Objective:** Verify the system properly acknowledges its limitations

**Limitation Test Queries:**
- "What is the best restaurant near the Golden Gate Bridge?" (irrelevant)
- "How do these structures compare to the Eiffel Tower?" (missing information)
- "What will these structures look like in 100 years?" (speculation)

**Expected Results:**
- Clear "insufficient information" responses
- No hallucination or speculation beyond provided context

### E.3: Multi-Language and Technical Term Testing
**Objective:** Test handling of technical terminology and varied expressions

**Technical Term Variations:**
- "ISS" vs "International Space Station"
- "ESB" vs "Empire State Building"
- "Hoover Dam" vs "Boulder Dam"
- Technical terms: "arch-gravity dam", "suspension bridge", "microgravity"

**Test Process:**
1. Use abbreviations and full names
2. Test technical vs layman terminology
3. Verify consistent retrieval regardless of term variation

---

## 📈 **CATEGORY F: PERFORMANCE AND OPTIMIZATION**

### F.1: Processing Speed Analysis
**Objective:** Understand the trade-offs between quality and speed

**Speed Test Matrix:**
- **Small Setup**: 1 document, 300 chunks, 3 results
- **Medium Setup**: 2 documents, 500 chunks, 5 results  
- **Large Setup**: 4 documents, 700 chunks, 8 results
- **Maximum Setup**: 4 documents, 1000 chunks, 15 results

**Measurements:**
- Document processing time
- Query response time
- Answer quality vs speed trade-offs

### F.2: Model Comparison Study
**Objective:** Compare different OpenAI models across various query types

**Test Matrix:**
- **Models**: GPT-3.5 Turbo, GPT-4, GPT-4 Turbo
- **Query Types**: Factual, analytical, comparative, creative
- **Metrics**: Accuracy, detail level, processing time, cost

**Sample Queries for Each Model:**
1. **Factual**: "What is the height of the Empire State Building?"
2. **Analytical**: "Analyze the engineering innovations in the Hoover Dam"
3. **Comparative**: "Compare the construction challenges of 1930s vs modern projects"
4. **Creative**: "Describe what makes these structures iconic"

### F.3: Parameter Preset Validation
**Objective:** Validate and potentially improve the built-in parameter presets

**Preset Testing:**
1. **Precise Mode**: Test with factual queries across all documents
2. **Balanced Mode**: Test with mixed query types
3. **Creative Mode**: Test with open-ended, interpretive queries

**Optimization Opportunity:**
- Document which presets work best for which query types
- Suggest improvements to preset configurations

---

## 🎓 **CATEGORY G: EDUCATIONAL AND DEMONSTRATION SCENARIOS**

### G.1: Progressive Complexity Demonstrations
**Objective:** Create a learning progression from simple to complex

**Level 1 - Basic Facts:**
- "When was the Empire State Building completed?"
- "How tall is the Golden Gate Bridge?"

**Level 2 - Comparisons:**
- "Which is taller: Empire State Building or Golden Gate Bridge towers?"
- "Compare construction times of 1930s projects"

**Level 3 - Analysis:**
- "What were the major engineering challenges of each project?"
- "How do these structures represent different engineering approaches?"

**Level 4 - Synthesis:**
- "How do these projects demonstrate the evolution of engineering?"
- "What lessons from these projects apply to modern construction?"

### G.2: Error Recovery and Troubleshooting Scenarios
**Objective:** Demonstrate systematic problem-solving approaches

**Common Problems and Solutions:**
1. **"Information not found"** → Increase chunk size and retrieval count
2. **"Irrelevant context"** → Reduce retrieval count, improve query phrasing
3. **"Incomplete answers"** → Increase chunk overlap, check document coverage
4. **"Inconsistent results"** → Lower temperature, enable query expansion

### G.3: Real-World Application Scenarios
**Objective:** Show how RAG applies to practical use cases

**Scenario Examples:**
1. **Research Assistant**: "Find all mentions of construction costs across documents"
2. **Fact Checker**: "Verify the height claims for each structure"
3. **Comparative Analyst**: "Create a timeline of all construction projects"
4. **Technical Writer**: "Extract key specifications for each structure"

---

## 🔧 **TESTING UTILITIES AND TOOLS**

### Quick Reference Commands
- **Reset Everything**: Clear All Data → Upload Documents → Set Parameters
- **Parameter Presets**: Use for consistent starting points
- **Export Results**: Save test results for analysis
- **Vector Explorer**: Examine chunk-level retrieval details

### Systematic Testing Approach
1. **Hypothesis**: What do you expect to happen?
2. **Test**: Run the specific test scenario
3. **Observe**: Note actual results vs expectations
4. **Analyze**: Why did you get these results?
5. **Document**: Record findings for future reference

### Performance Benchmarks
- **Good Retrieval**: Similarity scores < 0.4, relevant content
- **Acceptable Speed**: < 5 seconds for query processing
- **Quality Threshold**: Accurate, complete answers with proper attribution

---

## 📋 **TESTING CHECKLIST**

### Before Each Test Session:
- [ ] Clear all previous data
- [ ] Set known parameter baseline
- [ ] Upload appropriate test documents
- [ ] Verify query expansion setting

### During Testing:
- [ ] Document parameter settings used
- [ ] Note query phrasing exactly
- [ ] Record retrieval scores and chunks
- [ ] Save interesting results

### After Testing:
- [ ] Export test results if valuable
- [ ] Document lessons learned
- [ ] Note optimal parameter combinations
- [ ] Plan follow-up tests

---

## 🎯 **CONCLUSION**

This comprehensive testing guide provides both structured core tests for demonstrations and extensive exploration opportunities for deep learning. The combination of four diverse technical documents (Empire State Building, Golden Gate Bridge, Hoover Dam, and International Space Station) creates rich testing scenarios across different engineering domains, time periods, and complexity levels.

**Key Testing Principles:**
- Start with core tests to understand basics
- Use extra testing to explore edge cases and optimizations
- Document findings for systematic improvement
- Focus on real-world applicability

**Remember:** The goal isn't just to test the system, but to understand how RAG works, when it excels, and how to optimize it for different use cases. Each test teaches you something valuable about retrieval-augmented generation!

---

*This guide will continue to evolve as we discover new testing scenarios and optimization opportunities. The systematic approach ensures consistent, reproducible results while building deep understanding of RAG system behavior.* 