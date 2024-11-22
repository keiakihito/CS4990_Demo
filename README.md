# **CS4990_Demo**
This repository contains example CUDA codes for **cuBLAS**, **cuSPARSE**, and **cuSOLVER**, organized under the `src` directory. Additionally, profiling examples using **NVIDIA Nsight Systems** (`nsys`) are included in the `profile` directory.

---

## **Presentation**
The corresponding presentation slides detailing the CUDA API calls can be accessed [here](https://livecsupomona-my.sharepoint.com/:p:/r/personal/kkatsumi_cpp_edu/_layouts/15/Doc.aspx?sourcedoc=%7B8764CE80-58DF-4629-A221-6B0B8E375A3C%7D&file=CUDA%20API%20calls%202.pptx&action=edit&mobileredirect=true&DefaultItemOpen=1&ct=1732289317949&wdOrigin=OFFICECOM-WEB.START.EDGEWORTH&cid=2eb4ff28-190b-45cb-944b-01dc23985c0a&wdPreviousSessionSrc=HarmonyWeb&wdPreviousSession=8caa5da0-ac4d-4334-8e0f-0106cad87060).

---

## **Directory Structure**
- `src/`: Contains example source files for cuBLAS, cuSPARSE, and cuSOLVER API calls.
- `profile/`: Includes Nsight Systems profiling examples and configurations.

---

## **How to Use**
1. **Compile**:
   - Use `nvcc` to compile the example source files. For example:
     ```bash
     nvcc -o example_exampleName example_exampleName.cu -lcublas -lcusolver -lcusparse
     ```

2. **Run**:
   - Execute the compiled binaries:
     ```bash
     ./example_exampleName
     ```

3. **Profiling**:
   - Use NVIDIA Nsight Systems to profile examples in the `profile` directory. A sample command:
     ```bash
     nsys profile -o profile_output ./example_exampleName
     ```

---

## **Dependencies**
- **CUDA Toolkit**: Ensure you have the appropriate CUDA Toolkit installed.
- **Libraries**:
  - cuBLAS
  - cuSPARSE
  - cuSOLVER
- **Nsight Systems**: For profiling examples.

---

## **Acknowledgments**
This repository was created as part of the **CS4990** course presentation, focusing on demonstrating efficient use of CUDA libraries for numerical computing.

---