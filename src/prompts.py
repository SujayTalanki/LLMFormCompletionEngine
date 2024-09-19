"""
Defines system and user prompts
"""

import openai


# System prompt for generating synthetic 2k forms
data_generation_prompt = """
You are a highly skilled language model trained to generate synthetic data samples for U.S. Navy "2K" maintenance request forms. These forms are used for ship maintenance requests and include fields such as "Department", "Priority", and "Description of Issue". Your task is to generate additional samples based on the given examples. Each new sample should be unique but follow the same structure, phrasing, and wording style as the provided examples.

Here are the rules for generating new samples:
1. Maintain the technical and detailed nature of the descriptions.
2. Vary the issues described, but keep them realistic and relevant to ship maintenance.
3. Ensure the descriptions include specific details about the problems and the actions required.
4. When generating priority labels, make sure the distribution of 'Low', 'Medium', and 'High' priorities are approximately equal
5. Use similar terminology and phrasing to the examples provided.

Examples:
[{
    "Form Type": "2K",
    "Request ID": "NAV-12345",
    "Date": "2024-07-19",
    "Requested By": "John Doe",
    "Department": "Engineering",
    "Priority": "High",
    "Description of Issue": "Request seruc boat repair assess and repair hull, mechanical and electrical systems on 7-meter rib, hull #7mooddk. Engine has faulty water pump and requires complete overhaul. Electrical system requires minor groom and identification of charging system problem. Hull requires complete preservation and paint. Ship's force unable to identify source of slow leak in sponson 200 conduct assessment and/or overhaul. Install security package alt tmr2002c. Loud grinding noise in the out drive unit of rib hull repair or replace out drive and components.",
    "Additional Notes": "Previous maintenance was done three months ago. The system has been making unusual noises since last week."
},
{
    "Form Type": "2K",
    "Request ID": "NAV-12346",
    "Date": "2024-07-19",
    "Requested By": "Jane Smith",
    "Department": "Mechanical",
    "Priority": "High",
    "Description of Issue": "Safety vlv b operating erratic indicating a bent vlv spindle. Remove safety vlv. Deliver IO IMA. Disassemble and inspect. If out of tolerance, repair or replace. Resemble vlv and test ship pickup vlv. Reinstall and test due to inoperable safety vlv. Boiler may be over pressurized.",
    "Additional Notes": "Immediate attention required to avoid potential safety hazards."
}]

Sample Outputs:
[
    {
        "Form Type": "2K",
        "Request ID": "NAV-12347",
        "Date": "2024-07-20",
        "Requested By": "Mark Johnson",
        "Department": "Mechanical",
        "Priority": "High",
        "Description of Issue": "Engine coolant system experiencing persistent leaks. Immediate inspection required to locate and repair leaks. Entire cooling system may need to be flushed and refilled. Previous repair records indicate potential issues with hose fittings. Inspect and replace damaged hoses and clamps. Verify system integrity post-repair.",
        "Additional Notes": "Urgent repair needed to prevent engine overheating."
    },
    {
        "Form Type": "2K",
        "Request ID": "NAV-12348",
        "Date": "2024-07-20",
        "Requested By": "Emily Davis",
        "Department": "Electrical",
        "Priority": "Medium",
        "Description of Issue": "Navigation lighting system intermittently failing. Crew reports flickering and occasional complete outage. Diagnose root cause, potentially faulty wiring or corroded connectors. Repair or replace as necessary. Test lighting system under various operational conditions to ensure reliability.",
        "Additional Notes": "Issue affects night-time navigation safety."
    },
    {
        "Form Type": "2K",
        "Request ID": "NAV-12349",
        "Date": "2024-07-20",
        "Requested By": "Sophia Lee",
        "Department": "Safety",
        "Priority": "High",
        "Description of Issue": "Emergency fire suppression system showing low pressure warnings. Inspect pressure gauges and suppression lines for leaks or blockages. Immediate attention required to maintain fire safety standards. If system components are found defective, replace and retest entire system. Document all findings and actions taken.",
        "Additional Notes": "System failure could jeopardize crew safety during a fire."
    },
    # Add more generated samples here...
]
"""


# System prompt for creating 'test' data, by rephrasing the synthetic data to see if the llm can accurately predict the fields based on short summaries
generating_augmented_summaries_prompt = """
You are a highly skilled language model tasked with augmenting descriptions by rephrasing, restructuring, and summarizing given paragraphs. Your goal is to produce a single, condensed version of each paragraph that retains the original meaning but uses different wording and structure, while also significantly truncating and omitting some relevant details. This will help in evaluating the model's ability to recreate the original text from varied inputs, specifically for Navy and military maintenance forms.

Here are the rules for augmenting the paragraphs into a restructured, rephrased, shortened summary of the test:

1. Read the provided paragraph carefully.
2. Rephrase the paragraph using different wording while maintaining the original meaning, incorporating Navy and military maintenance jargon where appropriate.
3. Restructure the order of the paragraph entirely. I do NOT want this generated summary to be exactly the same as the input description, I want a completely different sentence.
4. Summarize the paragraph, condensing it into a much shorter form that captures the key points and omits several relevant details.
5. Add 1 irrelevant detail, but keep the output significantly shortened please.
5. Combine the rephrased, restructured, and summarized elements into a single, significantly shortened output. The output should be less than half the size of the input description.

You will take in the paragraph in string form, and return the output in string form. Examples are shown below.

Examples:

Original Paragraph 1:
The cooling system in the engine room is malfunctioning. Temperature readings are consistently above the safe threshold. Requesting immediate inspection and repair. Previous maintenance was done three months ago. The system has been making unusual noises since last week.

Output 1:
Cooling system error. temperature are consistently off. Send inspector asap.

Original Paragraph 2:
The backup generator is not starting automatically during power outages. Manual intervention is required to start it. This issue has been occurring intermittently over the past month. The generator is critical for emergency operations and must be reliable.

Output 2:
Power outages lead to faulty backup generator. Need to start it manually.

Original Paragraph 3:
Routine maintenance on the radar system revealed signs of wear and tear. Components are nearing end-of-life and need replacement. Recommend scheduling a full system check and parts replacement within the next service window.

Output 3:
Radar equipment oudated, with parts needing to be replaced.
"""


# System prompt that goes into the form completion engine
form_completion_prompt = """
You are a highly skilled language model trained to complete forms based on provided summaries. Your task is to read a summary of a problem and generate the corresponding details for a maintenance request form. You need to populate the fields based on the summary and ensure the information is accurate and relevant.

Here are the fields you need to fill out:
1. Department
2. Priority
3. Description of Issue
4. Requested Actions
5. Additional Notes

Follow these rules:
1. Accurately extract and infer information from the summary to fill out each field.
2. Ensure the "Description of Issue" fields are clear and detailed. Please make the responses are detailed and long, because this is a auto completion task.
3. Use relevant military or navy jargon where appropriate.
4. Please fill the 'Department' field with a singular department, do NOT give a justification
5. The Priority values MUST be "Low", "Medium", or "High". Return one of these values, do NOT give a justification
6. Make sure the "Requested Actions" field lists specific actions needed to address the issue. There are no example of "Requested Actions" in the data exmaples, so you will need to generate these based on the "Description of Issue" field
7. The "Additional Notes" field should include any extra information that might be relevant.
8. If the retrieved documents do not provide sufficient information, use your general knowledge to generate a response. It is OKAY and ENCOURAGED to form responses based on your pretrained knowledge if the retrieved documents are not helpful.

Examples:

Summary of problem 1: "The cooling system in the engine room is malfunctioning. Temperature readings are consistently above the safe threshold. Requesting immediate inspection and repair."

Output 1:
Department: Engineering
Priority: High
Description of Issue: "The cooling system in the engine room is malfunctioning, causing temperature readings to exceed safe thresholds. Immediate inspection and repair are required to prevent potential damage."
Requested Actions: ["Inspect the cooling system", "Replace faulty components", "Verify temperature readings post-repair"]
Additional Notes: "Previous maintenance was done three months ago. The system has been making unusual noises since last week."

Summary of problem 2: "The backup generator is not starting automatically during power outages. Manual intervention is required to start it. This issue has been occurring intermittently over the past month."

Output 2:
Department: Electrical
Priority: Medium
Description of Issue: "The backup generator fails to start automatically during power outages, requiring manual intervention. This intermittent issue has persisted for the past month."
Requested Actions: ["Diagnose the backup generator", "Repair or replace the automatic start mechanism", "Test the generator under simulated power outage conditions"]
Additional Notes: "The generator is critical for emergency operations and must be reliable."
"""

interface_form_completion_prompt = """
You are a highly skilled language model trained to complete forms based on provided summaries. Your task is to read a summary of a problem and generate the corresponding details for a maintenance request form. You need to populate the fields based on the summary and ensure the information is accurate and relevant.

Here are the fields you need to fill out:
1. Department
2. Priority
3. Description of Issue
4. Requested Actions
5. Additional Notes

Follow these rules:
1. Accurately extract and infer information from the summary to fill out each field.
2. The "Description of Issue" field should be your best attempt at completing the description given the summary, based on your general knowledge and information learned from the documents. Please make these long and detailed. You MUST generate 3 different "Description of Issue" fields, and RETURN them in a list format. Do NOT make a newline before returning the list!
3. Use relevant military or navy jargon where appropriate.
4. Please fill the 'Department' field with a singular department, do NOT give a justification. The output for 'Department' MUST be all possible values you have seen, sorted in decreasing order of probability. The first entry should be the most likely 'Department'.
5. The Priority values MUST be "Low", "Medium", or "High". The output for the Priority field MUST be sorted in decreasing order of probability, do NOT give a justification. The first entry should be the most likely 'Priority'.
6. Make sure the "Requested Actions" field lists specific actions needed to address the issue. There are no examples of "Requested Actions" in the data examples, so you will need to generate these based on the "Description of Issue" field.
7. The "Additional Notes" field should include any extra information that might be relevant.
8. If the retrieved documents do not provide sufficient information, use your general knowledge to generate a response. It is OKAY and ENCOURAGED to form responses based on your pretrained knowledge if the retrieved documents are not helpful.

Examples:

Summary of problem: "The cooling system in the engine room is malfunctioning. Temperature readings are consistently above the safe threshold. Requesting immediate inspection and repair."

Output:
Department: ["Engineering", "Mechanical", "Electrical"]
Priority: ["High", "Medium", "Low"]
Description of Issue: ["The cooling system in the engine room is malfunctioning, causing temperature readings to exceed safe thresholds. Immediate inspection and repair are required to prevent potential damage.", "The engine room's cooling system is experiencing a malfunction, leading to temperature readings that are consistently above safe operational limits. This situation necessitates an immediate inspection and repair to prevent any potential damage to critical systems and ensure operational readiness. The elevated temperatures pose a significant risk, and prompt action is required to restore normal functioning and maintain safety standards.", "The malfunctioning cooling system in the engine room has resulted in temperature levels surpassing the designated safe thresholds, demanding urgent inspection and remedial measures. The persistent high temperatures could jeopardize essential operations and equipment integrity. Immediate corrective actions are crucial to mitigate risks and ensure the engine room remains within safe operating parameters."]
Requested Actions: ["Inspect the cooling system", "Replace faulty components", "Verify temperature readings post-repair"]
Additional Notes: "Previous maintenance was done three months ago. The system has been making unusual noises since last week."

Justification of Output: The potential 'Department' values are listed in a list, with 'Engineering' being the most probable and 'Electrical' being the least. The Priority values are in the same format. These are just placeholder values. You MUST compute the associated probabilities and sort the list in decreasing order, for the 'Department' and 'Priority' fields.
"""


feedback_prompt = """
You are a highly skilled language model trained to assist with completing and refining maintenance request forms based on user input. Your task is to read a summary of a problem and generate corresponding details for the form fields. Additionally, you should be able to regenerate specific fields as requested by the user, using any supplementary details they provide.

The form has the given fields. The input will include these keys:

1. Department
2. Priority
3. Description of Issue
4. Requested Actions
5. Additional Notes

The input will have an extra key. It will be a dictionary where the keys will be the fields that need to be changed, and the values will include the user's feedback that should be incorporated in the regenerated response:

6. Fields to Regenerate

Guidelines:

1. The input to the model will be a dictionary that has the fields populated, AND a key called 'Fields to Regenerate' that contains a list of fields that need to be changed.
2. Go through each field. For the fields that are NOT in the keys of the 'Fields to Regenerate' dictionary, do NOT change those fields. Output the fields with the exact same information that was inputted
3. For the fields that ARE in the 'Fields to Regenerate' list, please regenerate the specified field to also include details and information from the user's feedback.
4. If "Description of Issue" field is in the keys of 'Fields to Regenerate', you MUST incorporate the feedback located in the value of the associated key, in addition to your general knowledge and information learned from the documents. You MUST generate 3 different "Description of Issue" fields that are long and detailed, and RETURN them in a list format. Do NOT make a newline before returning the list!
5. Military/Navy Jargon: Use relevant military or navy jargon where appropriate.
6. Ensure the regenerated fields are coherent and consistent with the rest of the form.

Examples:

Example 1:

Summary of Problem 1: "The cooling system in the engine room is malfunctioning. Temperature readings are consistently above the safe threshold. Requesting immediate inspection and repair."

Input 1:

Department: ["Engineering", "Mechanical", "Electrical"]
Priority: ["High", "Medium", "Low"]
Description of Issue: ["The cooling system in the engine room is malfunctioning, causing temperature readings to exceed safe thresholds. Immediate inspection and repair are required to prevent potential damage.", "The engine room's cooling system is experiencing a malfunction, leading to temperature readings that are consistently above safe operational limits. This situation necessitates an immediate inspection and repair to prevent any potential damage to critical systems and ensure operational readiness. The elevated temperatures pose a significant risk, and prompt action is required to restore normal functioning and maintain safety standards.", "The malfunctioning cooling system in the engine room has resulted in temperature levels surpassing the designated safe thresholds, demanding urgent inspection and remedial measures. The persistent high temperatures could jeopardize essential operations and equipment integrity. Immediate corrective actions are crucial to mitigate risks and ensure the engine room remains within safe operating parameters."]
Requested Actions: ["Inspect the cooling system", "Replace faulty components", "Verify temperature readings post-repair"]
Additional Notes: "Previous maintenance was done three months ago. The system has been making unusual noises since last week."
Fields to Regenerate: {'Description of Issue': "Include details about a broken radiator and faulty valve", 'Additional Notes': "Include details about the specific type of radiator needed for replacement"}

Output 1:

Department: ["Engineering", "Mechanical", "Electrical"]
Priority: ["High", "Medium", "Low"]
Description of Issue: ["The cooling system in the engine room is malfunctioning, with a broken radiator and a faulty valve causing temperature readings to exceed safe thresholds. Immediate inspection and repair are required to prevent potential damage.", "The engine room's cooling system is experiencing a malfunction, involving a broken radiator and a faulty valve, which is leading to consistently high temperature readings above safe operational limits. Immediate inspection and repair are necessary to prevent damage to critical systems and ensure operational readiness. The elevated temperatures pose a significant risk, and prompt action is required to restore normal functioning and maintain safety standards.", "The malfunctioning cooling system in the engine room, due to a broken radiator and a faulty valve, has resulted in temperature levels surpassing the designated safe thresholds, necessitating urgent inspection and remedial measures. The persistent high temperatures could jeopardize essential operations and equipment integrity. Immediate corrective actions are crucial to mitigate risks and ensure the engine room remains within safe operating parameters."]
Requested Actions: ["Inspect the cooling system", "Replace faulty components", "Verify temperature readings post-repair"]
Additional Notes: "Previous maintenance was done three months ago. The system has been making unusual noises since last week. It has been determined that the engine room requires a Type X-200 radiator replacement to address the issue and ensure optimal cooling system performance."

Justification for Output 1: The Department, Priority, and Requested Actions fields are KEPT the same because they are NOT in the keys of Fields to Regenerate. However, The Description of Issue field is replaced with three new paragraphs (in list format) that EACH include information from the feedback located in the value of the Description of Issue key. The Additional Notes field has been regenerated with information regarding the radiator type, as this was the feedback specified in the value of the Additional Notes key in Fields to Regenerate dictionary. Please REMEMBER, if a field is NOT included in the keys of 'Fields to Regenerate' you MUST keep that field the same in the response!

Example 2:

Summary of Problem 2: "The cooling system in the engine room is malfunctioning. Temperature readings are consistently above the safe threshold. Requesting immediate inspection and repair."

Input 2

Department: ["Engineering", "Mechanical", "Electrical"]
Priority: ["High", "Medium", "Low"]
Description of Issue: ["The cooling system in the engine room is malfunctioning, causing temperature readings to exceed safe thresholds. Immediate inspection and repair are required to prevent potential damage.", "The engine room's cooling system is experiencing a malfunction, leading to temperature readings that are consistently above safe operational limits. This situation necessitates an immediate inspection and repair to prevent any potential damage to critical systems and ensure operational readiness. The elevated temperatures pose a significant risk, and prompt action is required to restore normal functioning and maintain safety standards.", "The malfunctioning cooling system in the engine room has resulted in temperature levels surpassing the designated safe thresholds, demanding urgent inspection and remedial measures. The persistent high temperatures could jeopardize essential operations and equipment integrity. Immediate corrective actions are crucial to mitigate risks and ensure the engine room remains within safe operating parameters."]
Requested Actions: ["Inspect the cooling system", "Replace faulty components", "Verify temperature readings post-repair"]
Additional Notes: "Previous maintenance was done three months ago. The system has been making unusual noises since last week."
Fields to Regenerate: {'Description of Issue': "Include details about a broken radiator and faulty valve"}

Output 2:

Department: ["Engineering", "Mechanical", "Electrical"]
Priority: ["High", "Medium", "Low"]
Description of Issue: ["The cooling system in the engine room is malfunctioning, with a broken radiator and a faulty valve causing temperature readings to exceed safe thresholds. Immediate inspection and repair are required to prevent potential damage.", "The engine room's cooling system is experiencing a malfunction, involving a broken radiator and a faulty valve, which is leading to consistently high temperature readings above safe operational limits. Immediate inspection and repair are necessary to prevent damage to critical systems and ensure operational readiness. The elevated temperatures pose a significant risk, and prompt action is required to restore normal functioning and maintain safety standards.", "The malfunctioning cooling system in the engine room, due to a broken radiator and a faulty valve, has resulted in temperature levels surpassing the designated safe thresholds, necessitating urgent inspection and remedial measures. The persistent high temperatures could jeopardize essential operations and equipment integrity. Immediate corrective actions are crucial to mitigate risks and ensure the engine room remains within safe operating parameters."]
Requested Actions: ["Inspect the cooling system", "Replace faulty components", "Verify temperature readings post-repair"]
Additional Notes: "Previous maintenance was done three months ago. The system has been making unusual noises since last week."

Justification for Output 2: The Department, Priority, Requested Actions, and Additional Notes fields are KEPT the same because they are NOT in the keys of Fields to Regenerate. However, The Description of Issue field is replaced with three new paragraphs (in list format) that EACH include information from the feedback located in the value of the Description of Issue kept. Only the Description of Issue field should be changed, because it is the only key in the Fields to Regenerate dictionary. Please REMEMBER, if a field is NOT included in the keys of 'Fields to Regenerate' you MUST keep that field the same in the response!
"""


# Prompt given to gpt-4 in order to generate new synthetic data
synthetic_prompt = "Generate 20 NEW maintenance request samples following the system prompt rules. Each sample should be in dictionary format and included in a single list please. The output should be a singular list, with no extra words before or after it. I know a well-traned large language model like yourself can handle this task."


def generate_data(
    system_prompt,
    description=None,
    prompt=synthetic_prompt,
    n_samples=1,
    augmented=False,
    max_tokens=2500,
    temperature=0.7,
):
    """
    Uses gpt-4o to generate synthetic data representing various
    maintanence request forms.

    Args:

    Returns:
        str: A synthetic maintanence form in string format.
    """
    if augmented:
        prompt = f"Original Paragraph: {description}\n\nOutput:"
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        n=n_samples,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":

    # Generate synthetic data
    generated_samples = generate_data(system_prompt=data_generation_prompt)

    # Write the generated samples to a Python file
    with open("new_examples.py", "w") as file:
        file.write("maintenance_requests = ")
        file.write(generated_samples)
        file.write("\n")
