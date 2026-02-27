import http.client
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List

def make_api_call(dashboard_id):
    conn = http.client.HTTPSConnection("commonservice.ipms247.com")
    payload = json.dumps({
        "dashboardUnkId": dashboard_id
    })
    
    headers = {
        'accept': 'application/json',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
        'cache-control': 'no-cache',
        'content-type': 'application/json',
        'cookie': 'YCSK=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE3NDk2Mjc1NDUsInBpZCI6IlJtSlFVbWRDWlVWNGNWSm5OM04xYkdsSVRIQjRabkUzWmxselkwc3ZNRkJSTlZCdmREWTBNRlpKTDI5dVExbDNTamxPU1VOc1ZERTRhVEZpWmpSd1JWZEZUbkp5VDFGME1HVkdlbWMxUzB0YU5uZHpXRWc1VVdOYVEwaHhLM0ZWWlZwbUt6aHZRVUpUZEdGVFR5OUNZbkpFTm05ME1GSTVhblZCUzFkTVNESlJlV3QwWjNCRGFrOVJjME5KUzBKWlpFY3piVnBpTldFMFFVRXlWV2hFUkM5R1RreExkWGRJT1dSSlpFeDRRMlpQZUZGU0syaFFhV3cwYkcxMVVVc3JRa1l5VXlzdllVVlJTR2hQVEhkWU9HZ3JRelpIY0VwemNGbFVRVmRXVDFOVVJYRmFWMU5PUWs4NE5EMD0ifQ.ZDCQ55N7AywGsu4nnCdZ3BJmca-dCyrTPpSoBoyL0g4',
        'pragma': 'no-cache',
        'priority': 'u=1, i',
        'referer': 'https://commonstatic.ipms247.com/',
        'sec-ch-ua': '"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
        'sec-fetch-dest': 'empty', 
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
    }
    
    try:
        print(f"\nMaking API call for dashboardUnkId: {dashboard_id}")
        conn.request("POST", "/SwaggerService/saveDatasetFromTrigger", payload, headers)
        res = conn.getresponse()
        data = res.read()
        response_text = data.decode("utf-8")
        
        if res.status == 200:
            print(f"✅ Successfully processed dashboardUnkId: {dashboard_id}")
            return True, response_text
        else:
            print(f"❌ Failed to process dashboardUnkId: {dashboard_id} (Status: {res.status})")
            return False, response_text
            
    except Exception as e:
        print(f"❌ Error for dashboardUnkId {dashboard_id}: {str(e)}")
        return False, str(e)
    finally:
        conn.close()

def process_dashboard(dashboard_id: int, index: int, total: int) -> Tuple[int, bool, str]:
    print(f"\n📊 Processing dashboard ID {dashboard_id} ({index}/{total})")
    success, response = make_api_call(dashboard_id)
    return dashboard_id, success, response

def main():
    # List of dashboard IDs to process
    dashboard_ids = [2, 3, 4, 5, 6]
    total_ids = len(dashboard_ids)
    successful_calls = 0
    failed_calls = 0
    results: List[Tuple[int, bool, str]] = []
    
    print(f"\n🚀 Starting parallel API calls for {total_ids} dashboard IDs...")
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        future_to_id = {
            executor.submit(process_dashboard, dashboard_id, idx + 1, total_ids): dashboard_id 
            for idx, dashboard_id in enumerate(dashboard_ids)
        }
        
        # Process completed tasks as they finish
        for future in as_completed(future_to_id):
            dashboard_id, success, response = future.result()
            results.append((dashboard_id, success, response))
            
            if success:
                successful_calls += 1
            else:
                failed_calls += 1
    
    # Sort results by dashboard ID for consistent output
    results.sort(key=lambda x: x[0])
    
    # Print final summary
    print("\n" + "="*50)
    print("📊 FINAL SUMMARY")
    print("="*50)
    print(f"Total dashboard IDs processed: {total_ids}")
    print(f"✅ Successful calls: {successful_calls}")
    print(f"❌ Failed calls: {failed_calls}")
    print(f"Success rate: {(successful_calls/total_ids)*100:.2f}%")
    print("\nDetailed Results:")
    for dashboard_id, success, response in results:
        status = "✅" if success else "❌"
        print(f"{status} Dashboard ID {dashboard_id}: {'Success' if success else 'Failed'}")
    print("="*50)

if __name__ == "__main__":
    main()