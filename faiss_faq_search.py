# pip install -U sentence-transformers faiss-cpu
"""
This FAQ bot uses semantic search instead of exact keyword matching. Each FAQ question is converted into a dense vector embedding by a multilingual SentenceTransformer model. The embedding represents the meaning of the sentence in a numerical form, so questions with similar meanings can be close to each other even when they use different words or different languages.
Before building the search index, all FAQ questions are encoded and normalized. After normalization, comparing vectors with inner product is equivalent to cosine similarity. The normalized FAQ embeddings are stored in a FAISS IndexFlatIP index, which performs exact inner-product search. When the user enters a query, the query is also encoded and normalized in the same way, then FAISS searches for the FAQ question with the highest similarity score.
The system returns the answer associated with the most similar FAQ question. In this implementation, only the FAQ questions are embedded and searched, while the answers are stored as values in a dictionary and returned after retrieval. The model determines semantic similarity, FAISS provides efficient vector search, and the dictionary maps the matched question back to its final answer.
"""
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# 1. Load the sentence encoder model.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)
# 2. FAQ dictionary.
# The key is the question, which is used for vectorization and retrieval.
# The value is the answer. Triple quotes are used to support line breaks and indentation.
FAQ_DICT = {
    'Can this medical FAQ replace a doctor?': """No. This FAQ is for general health education only. It cannot diagnose disease, interpret your personal test results, prescribe treatment, or replace care from a licensed clinician.
The same symptom can have many causes. For example, fever may result from a viral infection, a bacterial infection, inflammation, medication reaction, autoimmune disease, heat illness, or another condition.
Medical decisions depend on age, medical history, physical examination, test results, pregnancy status, allergies, current medications, and local clinical guidelines.
Seek urgent medical care for warning signs such as chest pain, trouble breathing, sudden weakness or numbness, confusion, seizure, fainting, severe allergic reaction, severe headache, persistent high fever, severe dehydration, or rapidly worsening symptoms.""",
    'When should someone seek urgent medical care?': """Urgent medical care is appropriate when symptoms suggest a potentially serious or rapidly progressing condition.
Important warning signs include difficulty breathing, chest pain, blue lips, confusion, fainting, seizure, severe dehydration, persistent high fever, severe abdominal pain, vomiting blood, blood in stool, very low urine output, sudden weakness, sudden numbness, facial drooping, trouble speaking, severe allergic reaction, or severe headache of sudden onset.
Infants, older adults, pregnant people, immunocompromised people, and people with heart disease, kidney disease, liver disease, diabetes, cancer, or major chronic illness may need medical attention earlier than otherwise healthy adults.
This FAQ can help explain concepts, but it should not be used to decide that a serious symptom is safe.""",
    'What is the difference between DNA and RNA?': """DNA and RNA are both nucleic acids, but they differ in structure, chemistry, and biological role.
DNA usually exists as a double-stranded molecule. It contains deoxyribose sugar and the bases adenine, thymine, cytosine, and guanine. Its main role is long-term storage of genetic information.
RNA is usually single-stranded. It contains ribose sugar and the bases adenine, uracil, cytosine, and guanine; uracil replaces thymine.
Functionally, DNA is like a stable information archive, while RNA often acts as a working copy or functional molecule. Messenger RNA carries information for protein synthesis, transfer RNA helps bring amino acids to ribosomes, and ribosomal RNA is part of the ribosome.""",
    'What is a gene?': """A gene is a functional segment of DNA that contains information used to produce a specific RNA molecule or protein.
Genes often include more than just protein-coding sequence. They may also include regulatory regions that influence when, where, and how strongly the gene is expressed.
Genes affect traits by influencing cellular structure, enzyme activity, signaling, metabolism, and development.
Many traits are not controlled by a single gene alone. They often result from interactions among multiple genes, gene regulation, environment, nutrition, and life history.""",
    'What is a protein?': """A protein is a biological macromolecule made from amino acids linked by peptide bonds.
Proteins perform many essential functions. They can act as enzymes, structural components, transport molecules, receptors, hormones, antibodies, channels, or signaling molecules.
A protein's function depends strongly on its amino acid sequence and three-dimensional structure. If the structure is altered by mutation, heat, pH, chemicals, or other factors, function may decrease or be lost.
Examples include hemoglobin, which helps transport oxygen; antibodies, which help immune defense; and amylase, which helps digest starch.""",
    'How are proteins made?': """Protein synthesis generally involves transcription and translation.
During transcription, a cell uses a DNA template to make messenger RNA. The messenger RNA carries genetic information from the gene.
During translation, ribosomes read the messenger RNA in groups of three bases called codons. Transfer RNAs bring matching amino acids, which are linked together to form a polypeptide chain.
After translation, the polypeptide may fold, be chemically modified, move to a specific cellular location, or join other subunits before becoming a functional protein.
A simple summary is: DNA stores the instructions, RNA carries a working copy, and ribosomes build the protein.""",
    'What does the cell membrane do?': """The cell membrane is the boundary that separates the inside of a cell from its external environment.
It is mainly made of a phospholipid bilayer with embedded proteins. This structure gives the membrane selective permeability, meaning some substances cross more easily than others.
Small nonpolar molecules such as oxygen and carbon dioxide can often diffuse through the membrane. Ions, glucose, amino acids, and many larger or charged molecules usually require transport proteins.
The membrane also participates in cell signaling, immune recognition, cell adhesion, endocytosis, exocytosis, and maintenance of internal conditions.""",
    'What is the difference between mitosis and meiosis?': """Mitosis and meiosis are both forms of cell division, but they serve different purposes and produce different outcomes.
Mitosis usually occurs in somatic cells for growth, repair, and tissue maintenance. One cell division produces two daughter cells that are genetically very similar to the original cell and usually have the same chromosome number.
Meiosis occurs during the formation of gametes such as sperm and egg cells. It includes two rounds of division and produces cells with half the usual chromosome number.
Meiosis also promotes genetic variation through recombination and independent assortment, which helps explain why offspring are genetically different from their parents and siblings.""",
    'What is an enzyme?': """An enzyme is a biological catalyst. Most enzymes are proteins, although some RNA molecules can also have catalytic activity.
Enzymes speed up chemical reactions by lowering activation energy. They do not change the final equilibrium of the reaction and are not permanently consumed during the reaction.
Many enzymes are highly specific, meaning they act on particular substrates or types of reactions.
Enzyme activity can be affected by temperature, pH, substrate concentration, inhibitors, activators, and cofactors. In the human body, many enzymes work best near normal body temperature and within a narrow pH range.""",
    'What is cellular respiration?': """Cellular respiration is the process by which cells extract usable energy from organic molecules, commonly glucose.
In aerobic respiration, glucose is broken down through glycolysis, the citric acid cycle, and oxidative phosphorylation. The overall process produces carbon dioxide, water, and ATP.
ATP is a major energy currency of the cell. It powers processes such as muscle contraction, active transport, biosynthesis, and cellular signaling.
When oxygen is limited, some cells can use anaerobic pathways such as fermentation, but these usually produce much less ATP than aerobic respiration.""",
    'What is photosynthesis?': """Photosynthesis is the process by which plants, algae, and some bacteria use light energy to make organic molecules.
In plants, photosynthesis occurs mainly in chloroplasts. Chlorophyll and other pigments absorb light energy, which helps convert carbon dioxide and water into sugars while releasing oxygen.
Photosynthesis supplies organic carbon for plant growth and provides the primary energy input for many ecosystems.
Over geological time, photosynthesis has also played a major role in shaping Earth's oxygen-rich atmosphere.""",
    'What is a mutation?': """A mutation is a change in the genetic material of a cell or organism.
Mutations can occur during DNA replication or result from radiation, chemicals, viruses, or other sources of DNA damage.
A mutation may occur in a somatic cell or in a germ cell. Germ-cell mutations can be passed to offspring, while somatic mutations generally affect only the individual.
The effects of mutations vary. Some have little or no effect, some contribute to disease, and some may be beneficial in a particular environment.
Mutations are also an important source of genetic variation in evolution.""",
    'What is the immune system?': """The immune system is the body's defense network for recognizing and responding to pathogens, abnormal cells, toxins, and foreign substances.
It includes physical barriers such as skin and mucous membranes, immune cells such as white blood cells, organs such as lymph nodes and spleen, antibodies, complement proteins, and signaling molecules.
Innate immunity responds quickly and broadly. Adaptive immunity is more specific and can develop memory after infection or vaccination.
A well-regulated immune response helps protect the body. Too little immune function can increase infection risk, while misdirected or excessive immune responses can contribute to allergy, inflammation, or autoimmune disease.""",
    'What is an antibody?': """An antibody is a protein produced by plasma cells, which develop from B cells.
Antibodies bind specific targets called antigens. Antigens may be parts of viruses, bacteria, toxins, allergens, or other foreign substances.
When antibodies bind antigens, they can help neutralize pathogens, mark them for destruction, activate other immune mechanisms, or block harmful interactions.
Different antibody classes have different roles. For example, IgG is often important in longer-term systemic immunity, IgA is important at mucosal surfaces, and IgE is involved in allergic responses and defense against some parasites.""",
    'What is the difference between an antiviral drug and an antibiotic?': """Antiviral drugs and antibiotics target different kinds of pathogens.
Antiviral drugs are used against viruses. They may interfere with viral entry into cells, genome replication, protein processing, assembly, release, or other virus-specific steps.
Antibiotics are used against bacteria. They may inhibit bacterial cell-wall synthesis, protein synthesis, DNA replication, folate metabolism, or other bacterial processes.
Viruses do not have the same cellular structures as bacteria, so antibiotics generally do not treat viral infections such as colds or influenza.
Choosing between an antiviral, an antibiotic, supportive care, or no medication requires clinical judgment and sometimes diagnostic testing.""",
    'What are neuraminidase inhibitors?': """Neuraminidase inhibitors are antiviral drugs used against influenza viruses.
Influenza viruses use a surface enzyme called neuraminidase to help newly formed viral particles leave infected cells and spread to other cells.
Neuraminidase inhibitors reduce this enzymatic activity, which can limit viral spread within the respiratory tract.
Common examples include oseltamivir, zanamivir, and peramivir. These drugs are not general cold medicines and do not treat all viral infections.
Whether they are appropriate depends on timing, symptoms, risk factors, suspected or confirmed influenza, contraindications, and clinician judgment.""",
    'Is oseltamivir an antibiotic?': """No. Oseltamivir is not an antibiotic. It is an antiviral drug used for influenza.
Oseltamivir is a neuraminidase inhibitor. It acts on influenza virus neuraminidase and can reduce spread of influenza viruses from infected cells to nearby cells.
Antibiotics act against bacteria, not influenza viruses.
Oseltamivir is not a general fever medicine, pain medicine, or immune booster. It may be considered when influenza is suspected or confirmed, especially in higher-risk situations, but use should follow medical guidance.""",
    'Why are influenza antivirals often time-sensitive?': """Influenza antivirals are often most useful when started early because influenza virus replication is usually most active near the beginning of illness.
Early treatment may reduce symptom duration and may be especially important for people at higher risk of complications, such as older adults, very young children, pregnant people, and people with certain chronic medical conditions.
However, time-sensitive does not mean self-treatment is always appropriate. Fever, cough, body aches, and fatigue can have many causes.
Severe symptoms, worsening illness, shortness of breath, chest pain, confusion, dehydration, or high-risk medical status should prompt medical evaluation.""",
    'Why can viruses become resistant to antiviral drugs?': """Viruses can become less sensitive to antiviral drugs through genetic changes.
When a viral mutation alters the drug's target or affects viral replication pathways, the drug may bind less effectively or work less well.
Drug pressure can favor survival of less-susceptible viral variants, especially when antivirals are used inappropriately or in settings where prolonged replication occurs.
Resistance does not necessarily mean all antiviral options are ineffective. Management depends on the virus, the resistance pattern, the patient's condition, and available alternatives.""",
    'How are vaccines different from antiviral drugs?': """Vaccines and antiviral drugs act at different stages.
Vaccines are mainly preventive. They train the immune system to recognize a pathogen or part of a pathogen before a serious infection occurs.
Antiviral drugs are generally used after infection begins, or in some cases after high-risk exposure, to interfere with viral replication or spread.
Vaccines can provide longer-lasting population and individual protection, while antivirals are usually more time-limited and situation-specific.
They are not interchangeable. For diseases such as influenza, vaccination remains a major prevention strategy, while antivirals may be used for treatment or targeted prevention in selected cases.""",
    'What is uric acid?': """Uric acid is a normal waste product made when the body breaks down purines.
Purines come from the body's own cells and from some foods and beverages, including organ meats, some seafood, certain meats, beer, and some sugar-sweetened drinks.
Most uric acid dissolves in the blood, passes through the kidneys, and leaves the body in urine.
High uric acid can occur when the body produces too much uric acid, the kidneys remove too little, or both.
High uric acid does not always cause symptoms, but persistent elevation may increase the risk of gout, uric acid kidney stones, and kidney-related complications in some people.""",
    'What is hyperuricemia?': """Hyperuricemia means an abnormally high level of uric acid in the blood.
It can result from increased uric acid production, reduced kidney excretion, diet, alcohol use, obesity, insulin resistance, kidney disease, medications, genetics, or a combination of factors.
Hyperuricemia is not the same as gout. Many people with high uric acid never develop gout symptoms.
The clinical meaning of a uric acid result depends on the number, whether it persists, symptoms, kidney function, history of stones, gout flares, cardiovascular and metabolic risk factors, and medications.
A single abnormal value should be interpreted with a clinician rather than treated automatically.""",
    'Does high uric acid always mean gout?': """No. High uric acid is an important risk factor for gout, but it does not always mean a person has gout.
Gout occurs when monosodium urate crystals form and trigger inflammation, usually in joints or surrounding tissues.
Some people have high uric acid without symptoms. Conversely, during an acute gout flare, a blood uric acid level may sometimes be normal or only mildly elevated.
Gout diagnosis depends on symptoms, examination, history, and sometimes joint-fluid crystal analysis or imaging.
A hot, swollen, very painful joint can also be caused by infection or other inflammatory diseases, so severe or unusual joint symptoms require medical evaluation.""",
    'What does a gout flare usually feel like?': """A gout flare often causes sudden, intense joint pain with redness, warmth, swelling, and marked tenderness.
The base of the big toe is a classic location, but gout can also affect the ankle, knee, foot, wrist, fingers, or elbow.
Pain may begin suddenly, often at night or early morning, and can be severe enough that light touch is uncomfortable.
Similar symptoms can occur with septic arthritis, trauma, pseudogout, rheumatoid arthritis, or other conditions.
A red, hot, swollen joint with fever, chills, severe illness, or first-time symptoms should be assessed urgently to exclude joint infection.""",
    'How is uric acid related to the kidneys?': """The kidneys play a major role in removing uric acid from the body.
If kidney function declines, uric acid excretion may decrease, which can raise blood uric acid levels.
High uric acid can also contribute to uric acid kidney stones in some people. These stones may cause flank pain, blood in urine, nausea, urinary symptoms, or obstruction.
The relationship between uric acid and chronic kidney disease is complex and should not be reduced to one lab value.
Evaluation may include blood creatinine, estimated glomerular filtration rate, urinalysis, urine protein, imaging, medication review, blood pressure, diabetes status, and clinical history.""",
    'What is creatinine?': """Creatinine is a waste product generated from normal muscle metabolism.
The kidneys remove creatinine from the blood, so blood creatinine is commonly used as one marker of kidney function.
Creatinine must be interpreted carefully because it is affected by age, sex, body size, muscle mass, diet, hydration, medications, and acute illness.
Clinicians often use creatinine to estimate glomerular filtration rate, called eGFR, which provides a broader estimate of kidney filtering function.
An abnormal creatinine result should be interpreted with medical context and may require repeat testing, urine testing, medication review, and evaluation for acute or chronic kidney disease.""",
    'What dietary habits are often discussed for high uric acid?': """Dietary changes for high uric acid often focus on reducing purine load, limiting alcohol, maintaining hydration, and improving metabolic health.
Commonly limited items include organ meats, large amounts of red meat, some seafood, concentrated meat broths, beer, liquor, and sugar-sweetened beverages.
Helpful general habits may include adequate water intake, weight management when appropriate, regular physical activity, and a balanced diet emphasizing vegetables, whole grains, and low-fat dairy if tolerated.
Diet alone may not be enough for people with recurrent gout flares, tophi, kidney stones, or kidney disease.
Any treatment plan should be individualized by a clinician, especially when medications or chronic diseases are involved.""",
    'What are the main parts of the nervous system?': """The nervous system is commonly divided into the central nervous system and the peripheral nervous system.
The central nervous system includes the brain and spinal cord. It processes information, integrates signals, supports consciousness and cognition, and helps coordinate movement and homeostasis.
The peripheral nervous system includes nerves outside the brain and spinal cord. It carries sensory information to the central nervous system and motor or autonomic instructions back to muscles, glands, and organs.
These systems work together continuously. For example, touching something hot activates sensory nerves, spinal pathways, brain processing, and motor responses that help withdraw the hand.""",
    'What is a neuron?': """A neuron is a specialized nerve cell that receives, processes, and transmits information.
A typical neuron has a cell body, dendrites, and an axon. Dendrites receive signals, the cell body integrates information, and the axon transmits signals to other cells.
Neurons communicate through electrical impulses and chemical signals. At synapses, neurotransmitters can carry signals from one neuron to another neuron, muscle cell, or gland cell.
Neural circuits underlie sensation, movement, memory, emotion, language, reflexes, and autonomic control.
Although neurons are central to nervous-system function, supporting cells called glia are also essential for insulation, metabolism, immune defense, and tissue maintenance.""",
    'What is the difference between the central and peripheral nervous systems?': """The central nervous system consists of the brain and spinal cord.
It acts as an information-processing and coordination center, integrating sensory input, generating motor output, and supporting functions such as thought, emotion, memory, language, and homeostasis.
The peripheral nervous system consists of nerves and ganglia outside the brain and spinal cord.
It connects the central nervous system with the skin, muscles, joints, organs, and glands.
A simple analogy is that the central nervous system is the command and processing center, while the peripheral nervous system is the communication network connecting that center to the body.""",
    'What is the autonomic nervous system?': """The autonomic nervous system is the part of the peripheral nervous system that helps regulate involuntary body functions.
It influences heart rate, blood pressure, digestion, sweating, pupil size, temperature regulation, bladder function, and many organ functions.
It is often described as having sympathetic and parasympathetic divisions. Sympathetic activity is associated with stress responses such as increased heart rate and pupil dilation. Parasympathetic activity is associated with rest, digestion, and conservation of energy.
Autonomic symptoms such as palpitations, dizziness on standing, sweating changes, bowel changes, or bladder symptoms can have many causes and should be interpreted clinically.""",
    'What can cause numbness or tingling in the hands or feet?': """Numbness or tingling can come from many causes involving nerves, blood vessels, metabolism, the spine, or the brain.
Temporary symptoms may occur from pressure on a nerve, such as sleeping on an arm or sitting in one position for too long.
Persistent, recurrent, spreading, or painful numbness may relate to peripheral neuropathy, diabetes, vitamin B12 deficiency, thyroid disease, kidney disease, alcohol-related nerve injury, medication effects, carpal tunnel syndrome, cervical or lumbar spine disease, or other neurologic conditions.
Sudden one-sided numbness or weakness, especially with facial drooping, speech difficulty, vision changes, confusion, severe headache, or trouble walking, may indicate stroke and requires emergency care.""",
    'Is every headache a brain disease?': """No. Many headaches are not caused by a dangerous brain disease.
Common causes include tension-type headache, migraine, dehydration, sleep deprivation, stress, fever, sinus inflammation, eye strain, caffeine withdrawal, or neck muscle tension.
However, some headache patterns require urgent evaluation. Red flags include sudden severe headache, the worst headache of one's life, headache with fever and neck stiffness, headache with confusion or seizure, headache after head injury, headache with weakness or speech problems, new headache during pregnancy or after delivery, new headache in an older adult, or progressively worsening headache.
A change in headache pattern should be interpreted in clinical context.""",
    'What is a stroke?': """A stroke is an acute problem in which part of the brain is injured because blood flow is blocked or because a blood vessel ruptures.
An ischemic stroke is caused by blocked blood flow, often from a clot. A hemorrhagic stroke is caused by bleeding in or around the brain.
Stroke can cause sudden weakness, numbness, facial drooping, speech difficulty, confusion, vision changes, dizziness, trouble walking, loss of coordination, severe headache, or altered consciousness.
Stroke is a medical emergency because brain injury can progress quickly. Immediate emergency response can improve the chance of effective treatment and reduce disability.""",
    'How can someone recognize common stroke warning signs?': """A common stroke recognition tool is FAST.
F stands for Face: one side of the face may droop or feel weak.
A stands for Arm: one arm may drift downward or be difficult to raise.
S stands for Speech: speech may be slurred, strange, or difficult to understand.
T stands for Time: if any of these signs appear suddenly, emergency services should be called immediately.
Other warning signs include sudden trouble seeing, sudden confusion, sudden severe headache, sudden dizziness, loss of balance, trouble walking, or sudden numbness or weakness on one side of the body.""",
    'What is an antibiotic?': """An antibiotic is a medicine used to treat bacterial infections.
Antibiotics may kill bacteria directly or inhibit their growth so the immune system can clear the infection.
Different antibiotic classes work in different ways. Some inhibit bacterial cell-wall synthesis, some inhibit protein synthesis, some interfere with DNA replication, and others affect folate metabolism or cell membranes.
Antibiotics are not general anti-inflammatory drugs and are not fever reducers.
Whether an antibiotic is needed depends on whether a bacterial infection is likely, where the infection is, how severe it is, which organisms are likely, local resistance patterns, allergies, pregnancy status, kidney or liver function, and other patient-specific factors.""",
    'Do antibiotics treat colds or influenza?': """Usually no. Most colds and influenza are caused by viruses, and antibiotics do not treat viral infections.
Antibiotics do not kill cold viruses or influenza viruses and generally do not shorten the course of a viral upper respiratory infection.
A person with a viral infection can sometimes develop a secondary bacterial infection, but that requires clinical evaluation rather than automatic antibiotic use.
Yellow or green mucus, cough duration, sore throat, or fever alone does not prove a bacterial infection.
Unnecessary antibiotics can cause side effects and contribute to antibiotic resistance.""",
    'What is antibiotic resistance?': """Antibiotic resistance occurs when bacteria are no longer inhibited or killed by an antibiotic that would previously have been effective.
Resistant infections can be harder to treat, may last longer, and may require alternative drugs that can be more toxic, more expensive, or less convenient.
Resistance can develop through mutation, acquisition of resistance genes, and selection pressure from antibiotic exposure.
Misuse increases risk. Examples include taking antibiotics for viral infections, using leftover antibiotics, sharing antibiotics, stopping or extending treatment without guidance, or using broad-spectrum agents when narrow treatment would be sufficient.
Appropriate antibiotic use protects both individual patients and public health.""",
    'Why should antibiotics not be used casually?': """Antibiotics should not be used casually because they have benefits only when the likely cause is bacterial and the selected drug matches the infection.
Unnecessary use can cause diarrhea, nausea, rash, yeast infections, allergic reactions, drug interactions, kidney or liver problems, and disruption of normal microbiota.
It also increases selection pressure for resistant bacteria.
Using the wrong antibiotic may delay correct diagnosis and treatment.
A clinician chooses antibiotics by considering infection site, severity, likely organisms, test results, resistance patterns, allergies, organ function, pregnancy status, age, and other medications.""",
    'What are broad-spectrum and narrow-spectrum antibiotics?': """A broad-spectrum antibiotic acts against a wider range of bacteria, while a narrow-spectrum antibiotic acts against a more limited group.
Broad-spectrum antibiotics can be useful when a serious infection is present and the exact organism is not yet known.
Once the organism and susceptibility pattern are known, clinicians may narrow therapy to reduce side effects, microbiome disruption, and resistance pressure.
Broad-spectrum does not mean better for every infection. The goal is the most appropriate effective therapy, not the strongest or newest drug.
Antibiotic selection should be based on clinical evidence and stewardship principles.""",
    'What are common side effects of antibiotics?': """Antibiotic side effects vary by drug and patient.
Common effects can include nausea, diarrhea, abdominal discomfort, loss of appetite, rash, yeast infection, or changes in normal bacterial flora.
Serious but less common reactions include severe allergic reactions, severe skin reactions, liver injury, kidney injury, heart-rhythm effects, tendon injury with certain antibiotics, hearing or balance toxicity with certain drugs, and Clostridioides difficile-associated diarrhea.
Seek urgent care for trouble breathing, swelling of the lips or throat, widespread hives, fainting, severe rash, bloody diarrhea, severe abdominal pain, jaundice, very low urine output, or severe weakness.
Patients should tell clinicians about allergies, pregnancy, kidney or liver disease, and all current medicines or supplements.""",
    'What are cephalosporins?': """Cephalosporins are a large class of beta-lactam antibiotics used for selected bacterial infections.
They work by disrupting synthesis of the bacterial cell wall. Bacteria need a properly formed cell wall for structural support, and susceptible bacteria can die when cell-wall synthesis is blocked.
Cephalosporins are grouped into generations, but generation does not mean that a later drug is always better. Different generations and individual drugs cover different bacteria and are used in different clinical settings.
Some cephalosporins are oral, while others are injectable.
They are not antiviral drugs and do not treat ordinary viral colds or influenza.""",
    'Are cephalosporins antibiotics or antiviral drugs?': """Cephalosporins are antibiotics. They are not antiviral drugs.
They target susceptible bacteria by interfering with bacterial cell-wall synthesis.
They do not directly kill viruses such as cold viruses, influenza viruses, or SARS-CoV-2.
A viral illness may sometimes be complicated by a bacterial infection, but antibiotic use should be based on clinical assessment rather than symptom labels alone.
Using cephalosporins without a bacterial indication can cause side effects and contribute to resistance.""",
    'What are examples of cephalosporins?': """Examples of cephalosporins include cephalexin, cefazolin, cefuroxime, cefaclor, cefixime, cefdinir, ceftriaxone, cefotaxime, ceftazidime, cefepime, ceftaroline, and others.
These drugs are not interchangeable. They differ in spectrum, route of administration, dosing schedule, tissue penetration, kidney adjustment requirements, adverse-effect profile, and usual clinical use.
Some are mostly used orally in outpatient settings, while others are used intravenously in hospitals.
An open-source FAQ should explain the class and mechanism, but it should not recommend a specific cephalosporin for a symptom without medical evaluation.""",
    'What is the difference between cephalosporin generations?': """Cephalosporins are often grouped into generations based on development history and general antibacterial spectrum.
First-generation agents often have stronger activity against many gram-positive bacteria and more limited gram-negative coverage.
Second-generation agents generally expand coverage against some gram-negative organisms.
Third-generation agents often have greater gram-negative coverage and include drugs used in more serious infections, though individual agents differ.
Fourth- and fifth-generation agents have specialized roles and should be used with careful clinical judgment.
Generation is a simplification. The right antibiotic depends on infection site, organism, susceptibility, severity, patient factors, and stewardship considerations, not on a simple 'higher generation is better' rule.""",
    'Are cephalosporins related to penicillin?': """Cephalosporins and penicillins are both beta-lactam antibiotics, meaning they share a beta-lactam structural feature and inhibit bacterial cell-wall synthesis.
They are different drug families, but their chemical and immunologic similarities matter for allergy assessment.
A history of penicillin allergy does not automatically mean every cephalosporin is forbidden, but the details are important.
Severe reactions such as anaphylaxis, throat swelling, severe blistering rash, or organ involvement require special caution.
Anyone with a history of serious beta-lactam allergy should discuss it with a clinician before using a cephalosporin or penicillin.""",
    'What can cephalosporin allergy look like?': """Cephalosporin allergy can range from mild to life-threatening.
Milder reactions may include rash, itching, hives, or flushing.
Severe reactions may include wheezing, trouble breathing, throat tightness, swelling of the lips or face, dizziness, low blood pressure, fainting, widespread hives, or anaphylaxis.
Rare severe skin reactions can involve blistering, peeling skin, mouth sores, eye pain, or fever with rash.
Emergency care is needed for breathing difficulty, throat swelling, fainting, rapidly spreading hives, severe rash, or signs of shock.
A suspected drug allergy should be documented carefully, including the drug name, timing, symptoms, and severity.""",
    'Can cephalosporins be taken with alcohol?': """Alcohol should generally be avoided or discussed with a clinician during antibiotic treatment.
Some cephalosporins, especially those with certain side chains, have been associated with disulfiram-like reactions when combined with alcohol. Symptoms can include flushing, nausea, vomiting, headache, rapid heartbeat, and low blood pressure.
Not every cephalosporin has the same risk, but infection itself and medication side effects can also make alcohol a poor choice.
Patients should follow the specific medication label and clinician or pharmacist advice.
Severe symptoms after combining alcohol and medication require medical attention.""",
    'Is a higher-generation cephalosporin always better?': """No. A higher-generation cephalosporin is not automatically better.
Different cephalosporins cover different bacteria, reach different body sites, and have different risks.
Using a broader or higher-generation antibiotic when it is not needed can increase resistance pressure and disrupt normal microbiota without improving care.
For some infections, an older, narrower antibiotic may be more appropriate than a newer, broader one.
The best antibiotic is the one that is effective for the likely or confirmed pathogen, reaches the infection site, is safe for the patient, and uses the narrowest reasonable spectrum.""",
    'Why might an antibiotic fail?': """An antibiotic may fail for several reasons.
The illness may not be bacterial, or the bacteria may be resistant to the selected antibiotic.
The infection site may not receive adequate drug levels, or an abscess, infected device, obstruction, stone, dead tissue, or foreign body may require drainage or another procedure.
The diagnosis may be incomplete or incorrect, the patient may have an immune problem, or another condition may be mimicking infection.
Treatment may also be affected by absorption problems, drug interactions, missed doses, kidney function, or severity of illness.
If symptoms worsen or do not improve as expected, medical reassessment is safer than switching antibiotics independently.""",
    'Does a high white blood cell count always mean bacterial infection?': """No. A high white blood cell count can occur with bacterial infection, but it is not specific.
It can also occur with viral infections, inflammation, stress, trauma, corticosteroid use, smoking, pregnancy, autoimmune disease, tissue injury, and blood disorders.
Some serious infections may occur without a high white blood cell count, especially in older adults, immunocompromised people, or severe illness.
Clinicians interpret white blood cell count together with differential count, symptoms, physical findings, C-reactive protein, procalcitonin when appropriate, cultures, imaging, and clinical course.
A lab value should not be used alone to decide whether antibiotics are needed.""",
    'What does C-reactive protein mean?': """C-reactive protein, or CRP, is a blood marker that often rises with inflammation.
CRP can increase with infection, autoimmune disease, tissue injury, surgery, trauma, some cancers, and other inflammatory states.
A high CRP does not automatically prove bacterial infection, and a low CRP does not exclude all serious disease.
CRP is most useful when interpreted with symptoms, examination, other laboratory tests, imaging, and changes over time.
It should not be used alone to decide whether antibiotics are necessary.""",
    'Should every fever be treated with antibiotics?': """No. Fever does not automatically mean a bacterial infection.
Many fevers are caused by viruses such as common cold viruses, influenza viruses, or other respiratory viruses. Fever can also occur with inflammatory disease, medication reactions, heat illness, autoimmune disease, cancer, or other causes.
Antibiotics are appropriate only for selected bacterial infections.
Initial fever care often focuses on hydration, rest, monitoring, and identifying warning signs.
Persistent high fever, severe symptoms, breathing trouble, confusion, dehydration, rash, stiff neck, seizure, or high-risk medical status should prompt medical evaluation.""",
    'How can medicines be used more safely?': """Safe medicine use begins with knowing the exact drug name, purpose, dose instructions, timing, duration, major warnings, and what to do if a dose is missed.
Patients should not share prescription medicines, use leftover antibiotics, combine medicines without checking interactions, or stop important medicines without medical advice.
Clinicians and pharmacists should be told about allergies, pregnancy, breastfeeding, kidney disease, liver disease, heart rhythm problems, current medications, supplements, and previous serious drug reactions.
Urgent care is needed for severe allergic symptoms, trouble breathing, fainting, severe rash, jaundice, very low urine output, confusion, or other serious new symptoms after starting a medicine.""",
    'What is the first law of thermodynamics?': """The first law of thermodynamics is the conservation of energy applied to thermodynamic systems.
It states that energy cannot be created or destroyed, only transferred or converted between forms such as heat, work, and internal energy.
A common sign convention is delta U equals Q minus W, where delta U is the change in internal energy, Q is heat added to the system, and W is work done by the system on the surroundings.
For example, when gas in a cylinder is heated, some energy may increase the gas's internal energy and some may be used to push a piston outward.
The first law is an accounting principle for energy changes, not a statement about the direction in which processes naturally occur.""",
    'What is the second law of thermodynamics?': """The second law of thermodynamics describes the direction and limitations of natural processes.
One common statement is that the total entropy of an isolated system does not decrease in a spontaneous process.
Heat flows spontaneously from a hotter object to a colder object, not the reverse, unless external work is supplied.
The second law also implies that no heat engine can convert all absorbed heat into useful work with perfect efficiency.
This is why a perpetual-motion machine of the second kind is impossible under ordinary thermodynamic principles.""",
    'What is the third law of thermodynamics?': """The third law of thermodynamics states that the entropy of a perfect crystal approaches zero as temperature approaches absolute zero.
Absolute zero is 0 kelvin, or about minus 273.15 degrees Celsius.
One important implication is that absolute zero cannot be reached by a finite number of physical steps.
The third law is important in low-temperature physics, chemistry, materials science, and calculations of absolute entropy.
Real materials may have defects, disorder, residual entropy, or quantum effects, so actual systems can be more complex than the ideal statement.""",
    'What is the difference between temperature and heat?': """Temperature and heat are related but different concepts.
Temperature is a state variable that indicates thermal condition and is related to the average microscopic kinetic energy of particles in many systems.
Heat is energy transferred because of a temperature difference. It is not something a body simply contains; it is energy in transit between systems.
For example, a large tub of warm water may transfer more heat than a small cup of hotter water because total heat transfer depends on mass, specific heat, and temperature change.
Temperature describes a state. Heat describes a mode of energy transfer.""",
    'What is entropy?': """Entropy is a thermodynamic state function associated with energy dispersal and the number of microscopic arrangements compatible with a macroscopic state.
In an isolated system, spontaneous processes tend to increase total entropy.
For example, heat spreading from a hot object to a cold object makes energy more dispersed and increases total entropy.
Entropy is sometimes loosely described as disorder, but that phrase can be misleading. A more precise idea is that entropy relates to how many microscopic states correspond to the observed macroscopic condition.
Entropy helps explain why many processes are irreversible in practice.""",
    'What is the ideal gas law?': """The ideal gas law is PV equals nRT.
P is pressure, V is volume, n is the amount of gas in moles, R is the gas constant, and T is absolute temperature in kelvin.
The ideal gas model assumes gas particles have negligible volume and no intermolecular forces except during collisions.
Many real gases behave approximately like ideal gases at relatively high temperature and low pressure.
At high pressure, low temperature, or near condensation, real gases can deviate significantly from the ideal gas law.""",
    'What is internal energy?': """Internal energy is the total microscopic energy contained within a system.
It includes contributions from molecular motion, vibration, rotation, electronic states, and intermolecular interactions, depending on the system.
Internal energy is a state function, meaning it depends on the state of the system rather than the path used to reach that state.
Heat and work are not state functions; they describe ways energy crosses a system boundary during a process.
Changes in internal energy can occur through heating, cooling, compression, expansion, chemical reaction, phase change, or other interactions.""",
    "What is Newton's first law?": """Newton's first law is the law of inertia.
It states that an object remains at rest or continues moving in a straight line at constant speed unless acted on by a net external force.
This means force is not required to maintain motion. Force is required to change motion.
In everyday life, moving objects often slow down because of friction, air resistance, or other external forces.
The first law is most directly applied in inertial reference frames, where objects with zero net force have zero acceleration.""",
    "What is Newton's second law?": """Newton's second law relates net force, mass, and acceleration.
It is commonly written as F equals m times a, where F is the net external force, m is mass, and a is acceleration.
The acceleration points in the direction of the net force. For a given mass, a larger net force produces a larger acceleration. For a given force, a larger mass produces a smaller acceleration.
The key word is net. Multiple forces may act on an object, and their vector sum determines the acceleration.
The law is central to solving many problems in classical mechanics.""",
    "What is Newton's third law?": """Newton's third law states that when one object exerts a force on a second object, the second object exerts a force of equal magnitude and opposite direction on the first object.
These forces act on different objects, so they do not cancel each other on a single object.
For example, when a person pushes backward on the ground while walking, the ground pushes forward on the person.
The action and reaction forces are equal in size, but the resulting accelerations can differ because the two objects may have different masses.""",
    'What is momentum conservation?': """Momentum is the product of mass and velocity. For a single object, p equals m times v.
The law of conservation of momentum states that the total momentum of a system remains constant if the net external force on the system is zero.
This principle is especially useful for analyzing collisions, explosions, recoil, and interactions between objects.
Momentum can be conserved even when kinetic energy is not. In an inelastic collision, total momentum may remain constant while some mechanical energy becomes heat, sound, deformation, or internal energy.
The system boundary must be chosen carefully when applying the law.""",
    'When is mechanical energy conserved?': """Mechanical energy is the sum of kinetic energy and potential energy.
Mechanical energy is conserved when only conservative forces do work, or when the net work done by nonconservative forces is zero.
Common conservative forces include gravity and the ideal spring force. Common nonconservative forces include friction and air resistance.
For example, if air resistance is ignored, a falling object converts gravitational potential energy into kinetic energy while total mechanical energy remains constant.
If friction is present, mechanical energy may decrease, but total energy is still conserved because energy is transformed into internal energy, sound, or other forms.""",
    'What is the relationship between work and energy?': """Work is energy transferred by a force acting through a displacement.
If a force has a component in the same direction as displacement, it does positive work. If it has a component opposite the displacement, it does negative work.
The work-energy theorem states that the net work done on an object equals the change in the object's kinetic energy.
For example, the friction force during braking does negative work on a car, reducing its kinetic energy and converting some mechanical energy into thermal energy.
Work depends on force, displacement, and the angle between them.""",
    'What is inertia?': """Inertia is the tendency of an object to resist changes in its state of motion.
All objects with mass have inertia. Greater mass means greater inertia, so more net force is required to produce the same acceleration.
For example, when a car stops suddenly, passengers tend to continue moving forward because their bodies resist the change in motion.
Inertia is not a separate force. It is a property of matter.
Seat belts provide a force that changes a passenger's motion more safely during sudden braking or collision.""",
    'What is centripetal force?': """Centripetal force is the net force directed toward the center of a curved or circular path.
It is not a new fundamental force. It is the inward component or net result of actual forces such as tension, gravity, normal force, or friction.
For a ball swung on a string, tension can provide the centripetal force. For a car turning on a flat road, static friction can provide it.
If the required centripetal force is not available, the object cannot maintain the same curved path.
The object then moves according to the net force acting on it, often tangent to the previous path at the instant constraint is lost.""",
    'What is oxidation?': """Oxidation is the loss of electrons or an increase in oxidation number.
Historically, oxidation often meant reaction with oxygen, such as rusting or combustion. Modern chemistry uses the broader electron-transfer definition.
For example, when sodium becomes sodium ion, it loses one electron and is oxidized.
Oxidation always occurs together with reduction, because electrons lost by one species are gained by another.
A useful way to identify oxidation is to compare oxidation numbers before and after a reaction.""",
    'What is reduction?': """Reduction is the gain of electrons or a decrease in oxidation number.
For example, when chlorine becomes chloride ion, it gains an electron and is reduced.
Reduction always occurs together with oxidation. One substance gains electrons while another substance loses electrons.
The species that is reduced acts as the oxidizing agent because it causes another species to be oxidized.
A useful way to identify reduction is to look for a decrease in oxidation number.""",
    'What is the difference between an oxidizing agent and a reducing agent?': """An oxidizing agent causes another substance to lose electrons. The oxidizing agent itself gains electrons and is reduced.
A reducing agent causes another substance to gain electrons. The reducing agent itself loses electrons and is oxidized.
A compact way to remember this is: the oxidizing agent is reduced, and the reducing agent is oxidized.
For example, when magnesium reacts with oxygen to form magnesium oxide, magnesium loses electrons and acts as the reducing agent, while oxygen gains electrons and acts as the oxidizing agent.
The labels describe roles in a particular reaction, not permanent properties in all contexts.""",
    'How can I tell whether a reaction is a redox reaction?': """A reaction is a redox reaction if oxidation and reduction occur, usually shown by changes in oxidation numbers.
If at least one element's oxidation number increases and another element's oxidation number decreases, the reaction is redox.
For example, Zn plus Cu2+ forming Zn2+ plus Cu is redox. Zinc goes from 0 to plus 2 and is oxidized; copper goes from plus 2 to 0 and is reduced.
If oxidation numbers do not change, the reaction is usually not redox. Many acid-base neutralization reactions are not redox.
Checking oxidation numbers is more reliable than looking only for oxygen, hydrogen, or charge changes.""",
    'Is rusting a redox reaction?': """Yes. Rusting is a redox process involving iron, oxygen, water, and often electrolytes.
Iron atoms lose electrons and are oxidized. Oxygen is reduced as part of the overall corrosion process.
The final rust material is a mixture of iron oxides and hydrated iron oxides rather than one single simple compound in many real environments.
Moisture, salts, acids, and damaged protective coatings can accelerate rusting.
Rust prevention strategies include keeping oxygen and water away from the metal, painting, oiling, galvanizing, using stainless alloys, cathodic protection, or sacrificial anodes.""",
    'Is combustion a redox reaction?': """Many combustion reactions are redox reactions.
In typical combustion, a fuel is oxidized and oxygen is reduced. The reaction often releases heat and light.
For example, when methane burns, carbon in methane is oxidized to carbon dioxide, while oxygen is reduced to water.
Combustion usually requires fuel, an oxidizer, and sufficient activation energy, often called the fire triangle.
Not every exothermic reaction is combustion, and not every redox reaction involves a flame.""",
    'What is an electrochemical reaction?': """An electrochemical reaction is a redox reaction connected to the movement of electrons through an external circuit or driven by electrical energy.
In a galvanic cell, a spontaneous redox reaction produces electrical energy.
In an electrolytic cell, electrical energy drives a nonspontaneous redox reaction.
Oxidation occurs at the anode, and reduction occurs at the cathode. This is true for both galvanic and electrolytic cells, although the signs of the electrodes differ.
Batteries, fuel cells, corrosion, electroplating, and electrolysis all involve electrochemical principles.""",
    'How is an acid-base reaction different from a redox reaction?': """Acid-base reactions and redox reactions involve different kinds of chemical change.
An acid-base reaction usually involves proton transfer or neutralization. For example, hydrochloric acid reacting with sodium hydroxide forms sodium chloride and water without changing oxidation numbers.
A redox reaction involves electron transfer or changes in oxidation numbers.
Some reactions may involve both acid-base and redox processes, especially in complex aqueous chemistry.
The most reliable distinction is to check whether oxidation numbers change.""",
}
# 3. Extract the FAQ question list.
# The indices returned by FAISS are mapped back to questions.
questions = list(FAQ_DICT.keys())
# 4. Preprocess FAQ question text.
# Only questions are vectorized here, not answers.
faq_passages = [f"passage: {question}" for question in questions]
# 5. Encode FAQ questions into vectors.
# After normalization, inner product similarity is equivalent to cosine similarity ranking.
question_embeddings = model.encode(
    faq_passages,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")
# 6. Create a FAISS exact inner product index.
dim = question_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(question_embeddings)
# 7. FAQ retrieval function.
# It always returns the most similar answer with top_k=1, regardless of the similarity score.
def search_faq(query: str, top_k: int = 1):
    query_text = f"query: {query}"
    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    scores, indices = index.search(query_embedding, top_k)
    best_score = float(scores[0][0])
    best_idx = int(indices[0][0])
    matched_question = questions[best_idx]
    answer = FAQ_DICT[matched_question]
    return {
        "query": query,
        "matched_question": matched_question,
        "answer": answer,
        "score": best_score
    }
# 8. Continuous command-line chat.
# Enter exit / quit / q to stop the program.
def chat_loop():
    print("FAQ Bot started. Enter exit / quit / q to stop.")
    print("-" * 60)
    while True:
        user_query = input("User: ").strip()
        if not user_query:
            continue
        if user_query.lower() in ["exit", "quit", "q"]:
            print("System: Chat ended.")
            break
        result = search_faq(user_query, top_k=1)
        print("System:", result["answer"])
        print(f"Matched question: {result['matched_question']}")
        print(f"Similarity score: {result['score']:.4f}")
        print("-" * 60)
# 9. Program entry point.
# Start the chat loop when this file is run directly.
chat_loop()
