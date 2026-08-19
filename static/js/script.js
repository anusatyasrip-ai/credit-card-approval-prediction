/**
 * Credit Card Approval Prediction System JavaScript Helpers
 */

function loadSampleApplicant(type) {
    if (type === 'approved') {
        document.getElementById('gender').value = 'Female';
        document.getElementById('education_level').value = 'Higher education';
        document.getElementById('income_type').value = 'Salary';
        document.getElementById('employment_status').value = 'Employed';
        document.getElementById('employment_years').value = '6.0';
        document.getElementById('annual_income').value = '78000';
        document.getElementById('existing_loan_balance').value = '5000';
        document.getElementById('credit_history_years').value = '9.5';
        document.getElementById('credit_inquiries').value = '1';
        document.getElementById('payment_status').value = 'No Past Due';
    } else if (type === 'rejected') {
        document.getElementById('gender').value = 'Male';
        document.getElementById('education_level').value = 'Lower secondary';
        document.getElementById('income_type').value = 'Commercial';
        document.getElementById('employment_status').value = 'Unemployed';
        document.getElementById('employment_years').value = '0.0';
        document.getElementById('annual_income').value = '21000';
        document.getElementById('existing_loan_balance').value = '38000';
        document.getElementById('credit_history_years').value = '1.5';
        document.getElementById('credit_inquiries').value = '5';
        document.getElementById('payment_status').value = 'Past Due';
    }
}

function validatePredictionForm() {
    const annualIncome = parseFloat(document.getElementById('annual_income').value);
    const empYears = parseFloat(document.getElementById('employment_years').value);
    const loanBal = parseFloat(document.getElementById('existing_loan_balance').value);
    const inquiries = parseInt(document.getElementById('credit_inquiries').value);
    const creditHistory = parseFloat(document.getElementById('credit_history_years').value);

    if (isNaN(annualIncome) || annualIncome < 0) {
        alert('Annual income cannot be negative.');
        return false;
    }
    if (isNaN(empYears) || empYears < 0) {
        alert('Employment duration cannot be negative.');
        return false;
    }
    if (isNaN(loanBal) || loanBal < 0) {
        alert('Existing loan balance cannot be negative.');
        return false;
    }
    if (isNaN(inquiries) || inquiries < 0) {
        alert('Credit inquiries cannot be negative.');
        return false;
    }
    if (isNaN(creditHistory) || creditHistory < 0) {
        alert('Credit history duration cannot be negative.');
        return false;
    }

    return true;
}
